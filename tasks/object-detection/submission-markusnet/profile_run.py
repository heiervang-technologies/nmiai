"""
Profiling wrapper for MarkusNet inference pipeline.
Instruments each stage with CUDA-synced timing.
"""
import argparse
import json
import time
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Import everything from run.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run import (
    MarkusNet, load_onnx_session, preprocess_image, decode_yolo_output,
    SCRIPT_DIR, YOLO_MODEL, MARKUSNET_CKPT, NUM_CLASSES, CROP_BATCH_SIZE,
    IMAGE_TOKEN_ID, QWEN_IMAGE_SIZE, VIS_SPATIAL_MERGE,
    CHAT_PREFIX_IDS, CHAT_SUFFIX_IDS,
)


class Timer:
    """CUDA-synced timer for accurate GPU profiling."""
    def __init__(self, device):
        self.device = device
        self.records = defaultdict(list)

    def start(self, name):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._current = name
        self._t0 = time.perf_counter()

    def stop(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t0
        self.records[self._current].append(elapsed)
        return elapsed

    def report(self):
        print("\n" + "=" * 70)
        print("PROFILING RESULTS")
        print("=" * 70)
        total = 0
        for name, times in self.records.items():
            s = sum(times)
            total += s
            count = len(times)
            avg = s / count if count > 0 else 0
            print(f"  {name:40s}  total={s:8.3f}s  count={count:4d}  avg={avg:.4f}s")
        print(f"  {'TOTAL':40s}  total={total:8.3f}s")
        print("=" * 70)
        return self.records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="/tmp/profiling_predictions.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    timer = Timer(device)

    # ===== Stage 1: Load YOLO =====
    timer.start("1_yolo_load")
    available_providers = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in available_providers
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    yolo_path = SCRIPT_DIR / YOLO_MODEL
    if not yolo_path.exists():
        for alt in ["yolo11x_v3.onnx", "yolo26x_v3.onnx"]:
            alt_path = SCRIPT_DIR / alt
            if alt_path.exists():
                yolo_path = alt_path
                break
    print(f"Loading YOLO from {yolo_path}")
    yolo_session, yolo_input_name, yolo_input_shape = load_onnx_session(yolo_path, providers)
    timer.stop()

    # ===== Stage 2: Load MarkusNet checkpoint =====
    timer.start("2_markusnet_init")
    model = MarkusNet()
    timer.stop()

    timer.start("3_nf4_dequant_and_load")
    model.load_checkpoint(str(SCRIPT_DIR / MARKUSNET_CKPT), device)
    timer.stop()

    timer.start("4_model_to_device")
    model = model.to(dtype).to(device)
    model.eval()
    timer.stop()

    # ===== Process images =====
    input_dir = Path(args.input)
    image_paths = sorted(
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg")) +
        list(input_dir.glob("*.png"))
    )
    print(f"Found {len(image_paths)} images")

    results = []
    total_crops = 0

    for img_idx, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        # ===== Stage 5: YOLO detection =====
        timer.start("5_yolo_preprocess")
        tensor, ratio, pad = preprocess_image(image_bgr, yolo_input_shape)
        timer.stop()

        timer.start("6_yolo_inference")
        outputs = yolo_session.run(None, {yolo_input_name: tensor})
        timer.stop()

        timer.start("7_yolo_postprocess")
        boxes, det_scores = decode_yolo_output(outputs, ratio, pad, image_bgr.shape[:2])
        timer.stop()

        num_dets = len(boxes)
        print(f"  Image {img_idx}: {img_path.name} -> {num_dets} detections")

        if num_dets == 0:
            continue

        # ===== Stage 6: Crop extraction =====
        timer.start("8_crop_extraction")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        crops = []
        for box in boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(image_pil.width, int(box[2]))
            y2 = min(image_pil.height, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(Image.new("RGB", (32, 32), (114, 114, 114)))
            else:
                crops.append(image_pil.crop((x1, y1, x2, y2)))
        timer.stop()

        total_crops += len(crops)

        # ===== Stage 7: MarkusNet classification (detailed) =====
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            with torch.inference_mode():
                for crop_idx, crop in enumerate(crops):
                    # Preprocess
                    timer.start("9_vision_preprocess")
                    pv, grid_thw = model._preprocess_image(crop, device)
                    timer.stop()

                    # Vision encoder
                    timer.start("10_vision_encoder")
                    vis_embeds = model.vision(pv, [grid_thw])
                    timer.stop()

                    # Build embeddings
                    timer.start("11_embed_scatter")
                    num_image_tokens = vis_embeds.shape[0]
                    t, h, w = grid_thw
                    llm_grid_h = h // VIS_SPATIAL_MERGE
                    llm_grid_w = w // VIS_SPATIAL_MERGE

                    prefix_ids = CHAT_PREFIX_IDS
                    suffix_ids = CHAT_SUFFIX_IDS
                    image_ids = [IMAGE_TOKEN_ID] * num_image_tokens
                    input_ids = torch.tensor(
                        [prefix_ids + image_ids + suffix_ids],
                        dtype=torch.long, device=device,
                    )
                    inputs_embeds = model.language.embed_tokens(input_ids)
                    image_mask = (input_ids == IMAGE_TOKEN_ID)
                    image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(
                        image_mask_3d, vis_embeds.to(inputs_embeds.dtype),
                    )
                    position_ids = model._build_position_ids(
                        len(prefix_ids), num_image_tokens, len(suffix_ids),
                        t, llm_grid_h, llm_grid_w, device,
                    )
                    timer.stop()

                    # Language model
                    timer.start("12_language_model")
                    hidden = model.language(inputs_embeds, position_ids=position_ids)
                    timer.stop()

                    # Classification head
                    timer.start("13_cls_head")
                    logits = model.cls_head(hidden)
                    timer.stop()

        # Combine scores (not profiled - trivial)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                cat_ids, cls_conf = model.classify_crops(crops[:0], device)  # dummy to avoid re-running
        # Actually compute from already-computed logits is trivial, skip for profiling

    print(f"\nTotal crops classified: {total_crops}")
    print(f"Images processed: {len(image_paths)}")

    # Extrapolation
    records = timer.report()

    # Extrapolate to 200 images
    n_images = len(image_paths)
    if n_images > 0 and total_crops > 0:
        avg_crops_per_image = total_crops / n_images
        target_images = 200
        target_crops = target_images * avg_crops_per_image

        print(f"\n{'='*70}")
        print(f"EXTRAPOLATION TO {target_images} IMAGES ({target_crops:.0f} crops)")
        print(f"{'='*70}")

        # One-time costs
        one_time = sum(sum(records[k]) for k in records if k.startswith(("1_", "2_", "3_", "4_")))
        print(f"  One-time (model loading):  {one_time:.3f}s")

        # Per-image costs
        per_image_keys = ["5_yolo_preprocess", "6_yolo_inference", "7_yolo_postprocess", "8_crop_extraction"]
        per_image = sum(sum(records[k]) for k in per_image_keys) / n_images
        print(f"  Per-image (YOLO + crops):  {per_image:.4f}s x {target_images} = {per_image * target_images:.3f}s")

        # Per-crop costs
        per_crop_keys = ["9_vision_preprocess", "10_vision_encoder", "11_embed_scatter",
                         "12_language_model", "13_cls_head"]
        per_crop = sum(sum(records[k]) for k in per_crop_keys) / total_crops if total_crops > 0 else 0
        print(f"  Per-crop (MarkusNet):      {per_crop:.4f}s x {target_crops:.0f} = {per_crop * target_crops:.3f}s")

        # Breakdown per crop
        for k in per_crop_keys:
            t = sum(records[k]) / total_crops if total_crops > 0 else 0
            print(f"    {k:35s}  {t:.4f}s/crop  ({t * target_crops:.1f}s total)")

        estimated_total = one_time + per_image * target_images + per_crop * target_crops
        print(f"\n  ESTIMATED TOTAL: {estimated_total:.1f}s (budget: 300s)")
        print(f"  {'FITS' if estimated_total < 250 else 'EXCEEDS'} 250s target")


if __name__ == "__main__":
    main()
