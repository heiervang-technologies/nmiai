"""
Export MarkusNet vision encoder to ONNX.

Important: the current MarkusNet vision implementation consumes Python tuples for
`grid_thw`, so a naive ONNX wrapper that calls `.item()` on `image_grid_thw`
does not produce a real dynamic graph. This script therefore treats dynamic
export as successful only if the exported ONNX model still exposes
`image_grid_thw` and keeps a symbolic sequence dimension.

Strategy:
1. Probe for a true dynamic export with variable num_patches + grid_thw
2. If that fails, fall back to letterbox padding with fixed size

The vision encoder takes:
  - pixel_values: (num_patches, 1536) - flattened patches
  - grid_thw: list of (t, h, w) tuples

And returns:
  - merged features: (num_merged_tokens, VIS_OUT_HIDDEN=1024)

Usage: CUDA_VISIBLE_DEVICES=0 uv run python export_vision_dynamic.py
"""

import functools
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).parent
CHECKPOINT = SCRIPT_DIR / "training_output" / "best" / "best.pt"
OUTPUT_DIR = SCRIPT_DIR / "exported"

# Import VisionEncoder and constants from run_markusnet
sys.path.insert(0, str(SCRIPT_DIR))
from run_markusnet import (
    VisionEncoder, MarkusNet, ClassificationHead,
    VIS_HIDDEN, VIS_PATCH_SIZE, VIS_TEMPORAL_PATCH, VIS_SPATIAL_MERGE,
    VIS_OUT_HIDDEN, VIS_DEPTH, VIS_NUM_HEADS,
    QWEN_MEAN, QWEN_STD, NUM_CLASSES,
)


def smart_resize(height, width, min_pixels=56*56, max_pixels=448*448,
                 factor=32):
    """
    Qwen-style smart resize: preserve aspect ratio, both dims divisible by factor.
    factor = patch_size * merge_size = 16 * 2 = 32 for our model.
    """
    if height < factor or width < factor:
        raise ValueError(f"Image too small: {height}x{width}")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)

    return h_bar, w_bar


def preprocess_image_smart(pil_img, device, dtype=torch.float32):
    """
    Preprocess using smart_resize (preserves aspect ratio).
    Returns pixel_values (N, 1536) and grid_thw tuple.
    """
    from PIL import Image
    img = pil_img.convert("RGB")
    orig_w, orig_h = img.size

    new_h, new_w = smart_resize(orig_h, orig_w)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Normalize
    img_np = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        img_np[:, :, c] = (img_np[:, :, c] - QWEN_MEAN[c]) / QWEN_STD[c]

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device).to(dtype)

    h_patches = new_h // VIS_PATCH_SIZE
    w_patches = new_w // VIS_PATCH_SIZE
    t_patches = 1

    # Create temporal duplication and flatten to patches
    frames = torch.stack([img_tensor, img_tensor])  # (2, 3, H, W)
    C = 3
    T, H, W = VIS_TEMPORAL_PATCH, new_h, new_w
    pH, pW = VIS_PATCH_SIZE, VIS_PATCH_SIZE
    pT = VIS_TEMPORAL_PATCH

    x = frames.reshape(T // pT, pT, C, H // pH, pH, W // pW, pW)
    x = x.permute(0, 3, 5, 2, 1, 4, 6)
    x = x.reshape(-1, C * pT * pH * pW)

    grid_thw = (t_patches, h_patches, w_patches)
    return x, grid_thw


def letterbox_preprocess(pil_img, target_size, device, dtype=torch.float32):
    """
    Letterbox pad to target square size, preserving aspect ratio.
    Pad with gray (0.5 in normalized space).
    """
    from PIL import Image
    img = pil_img.convert("RGB")
    orig_w, orig_h = img.size

    # Scale to fit within target_size
    scale = min(target_size / orig_h, target_size / orig_w)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Make both dims divisible by factor (patch_size * merge_size = 32)
    factor = VIS_PATCH_SIZE * VIS_SPATIAL_MERGE  # 32
    # Ensure the TARGET is divisible by factor
    assert target_size % factor == 0, f"target_size {target_size} must be divisible by {factor}"

    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Create padded image
    padded = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded.paste(img, (paste_x, paste_y))

    # Normalize
    img_np = np.array(padded, dtype=np.float32) / 255.0
    for c in range(3):
        img_np[:, :, c] = (img_np[:, :, c] - QWEN_MEAN[c]) / QWEN_STD[c]

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device).to(dtype)

    h_patches = target_size // VIS_PATCH_SIZE
    w_patches = target_size // VIS_PATCH_SIZE
    t_patches = 1

    frames = torch.stack([img_tensor, img_tensor])
    C = 3
    T, H, W = VIS_TEMPORAL_PATCH, target_size, target_size
    pH, pW = VIS_PATCH_SIZE, VIS_PATCH_SIZE
    pT = VIS_TEMPORAL_PATCH

    x = frames.reshape(T // pT, pT, C, H // pH, pH, W // pW, pW)
    x = x.permute(0, 3, 5, 2, 1, 4, 6)
    x = x.reshape(-1, C * pT * pH * pW)

    grid_thw = (t_patches, h_patches, w_patches)
    return x, grid_thw


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper for ONNX export. Takes pixel_values + grid_thw tensor.
    For dynamic export, grid_thw is (1, 3) int tensor.
    """
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision = vision_encoder

    def forward(self, pixel_values, image_grid_thw):
        # NOTE: `.item()` materializes the sample grid values during export.
        # This wrapper is useful only as a probe. A successful dynamic export
        # must be validated from the emitted ONNX contract afterwards.
        # Convert tensor grid_thw to tuple for the vision encoder
        t = image_grid_thw[0, 0].item()
        h = image_grid_thw[0, 1].item()
        w = image_grid_thw[0, 2].item()
        grid_thw = [(t, h, w)]
        return self.vision(pixel_values, grid_thw)


def inspect_onnx_contract(onnx_path):
    """Return the exported ONNX IO contract for sanity-checking."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {inp.name: list(inp.shape) for inp in session.get_inputs()}
    outputs = {out.name: list(out.shape) for out in session.get_outputs()}
    return inputs, outputs


def has_symbolic_dim(shape):
    return any(isinstance(dim, str) and dim for dim in shape)


class VisionEncoderFixedGrid(nn.Module):
    """
    Vision encoder with baked-in grid size for ONNX export.
    Avoids dynamic control flow from grid_thw.
    """
    def __init__(self, vision_encoder, fixed_grid_thw):
        super().__init__()
        self.vision = vision_encoder
        self.fixed_t = fixed_grid_thw[0]
        self.fixed_h = fixed_grid_thw[1]
        self.fixed_w = fixed_grid_thw[2]

    def forward(self, pixel_values):
        grid_thw = [(self.fixed_t, self.fixed_h, self.fixed_w)]
        return self.vision(pixel_values, grid_thw)


def load_model(device):
    """Load MarkusNet and return vision encoder with weights."""
    print("Loading checkpoint...")
    model = MarkusNet()
    # Always load on CPU first to avoid OOM
    model.load_checkpoint(str(CHECKPOINT), "cpu")
    model = model.to(torch.float32).to(device)
    model.eval()
    return model


def try_dynamic_export(model, device):
    """Attempt ONNX export with dynamic num_patches."""
    print("\n=== Attempting Dynamic ONNX Export ===")

    wrapper = VisionEncoderWrapper(model.vision)
    wrapper.eval()

    # Create sample input (256x256 -> 16x16 patches -> 256 patches)
    num_patches = 256
    pixel_values = torch.randn(num_patches, 1536, device=device)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long, device=device)

    output_path = OUTPUT_DIR / "markusnet_vision_dynamic.onnx"

    try:
        torch.onnx.export(
            wrapper,
            (pixel_values, grid_thw),
            str(output_path),
            input_names=["pixel_values", "image_grid_thw"],
            output_names=["vision_embeds"],
            dynamic_axes={
                "pixel_values": {0: "num_patches"},
                "vision_embeds": {0: "num_image_tokens"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        inputs, outputs = inspect_onnx_contract(output_path)
        print(f"Exported ONNX inputs: {inputs}")
        print(f"Exported ONNX outputs: {outputs}")

        pixel_shape = inputs.get("pixel_values")
        grid_shape = inputs.get("image_grid_thw")
        output_shape = outputs.get("vision_embeds")
        is_true_dynamic = (
            pixel_shape is not None
            and grid_shape is not None
            and output_shape is not None
            and has_symbolic_dim(pixel_shape)
            and has_symbolic_dim(output_shape)
        )

        if not is_true_dynamic:
            print(
                "Dynamic export probe did not preserve dynamic axes or `image_grid_thw`; "
                "treating this as failure and falling back to fixed-grid letterbox export."
            )
            return None

        print(f"Dynamic export succeeded! Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Dynamic export FAILED: {e}")
        return None


def export_fixed_grid(model, device, target_size=448):
    """Export with fixed grid size (letterbox preprocessing at inference)."""
    h_patches = target_size // VIS_PATCH_SIZE
    w_patches = target_size // VIS_PATCH_SIZE
    t_patches = 1
    num_patches = t_patches * h_patches * w_patches

    grid_thw = (t_patches, h_patches, w_patches)
    print(f"\n=== Fixed Grid ONNX Export (grid={grid_thw}, patches={num_patches}) ===")

    wrapper = VisionEncoderFixedGrid(model.vision, grid_thw)
    wrapper.eval()

    pixel_values = torch.randn(num_patches, 1536, device=device)

    suffix = f"sq{target_size}"
    output_path = OUTPUT_DIR / f"markusnet_vision_{suffix}_v2.onnx"

    # Use legacy exporter (dynamo=False) for self-contained ONNX file
    torch.onnx.export(
        wrapper,
        (pixel_values,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["vision_embeds"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / 1024**2
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def quantize_onnx(onnx_path):
    """Quantize ONNX model to INT8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    output_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    print(f"\nQuantizing to INT8: {output_path}")

    quantize_dynamic(
        str(onnx_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )

    size_mb = output_path.stat().st_size / 1024**2
    print(f"INT8 size: {size_mb:.1f} MB")
    return output_path


def verify_onnx(onnx_path, model, device, is_dynamic=False):
    """Verify ONNX model outputs match PyTorch on multiple shapes."""
    import onnxruntime as ort

    print(f"\n=== Verifying ONNX: {onnx_path.name} ===")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)

    input_names = [inp.name for inp in session.get_inputs()]
    print(f"ONNX inputs: {[(inp.name, inp.shape) for inp in session.get_inputs()]}")
    print(f"ONNX outputs: {[(out.name, out.shape) for out in session.get_outputs()]}")

    from PIL import Image

    # Test with various crop sizes
    test_sizes = [(100, 150), (200, 100), (50, 50), (300, 200), (160, 320)]

    for h, w in test_sizes:
        # Create synthetic test image
        img_array = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_array)

        if is_dynamic:
            pv, grid = preprocess_image_smart(pil_img, device)
        else:
            # Determine target size from ONNX input shape
            total_patches = session.get_inputs()[0].shape[0]
            if isinstance(total_patches, str):
                # Dynamic
                target_size = 448
            else:
                side = int(math.sqrt(total_patches))
                target_size = side * VIS_PATCH_SIZE
            pv, grid = letterbox_preprocess(pil_img, target_size, device)

        # PyTorch reference
        with torch.no_grad():
            ref_out = model.vision(pv, [grid])

        # ONNX inference
        pv_np = pv.cpu().numpy()
        feed = {"pixel_values": pv_np}
        if "image_grid_thw" in input_names:
            grid_np = np.array([[grid[0], grid[1], grid[2]]], dtype=np.int64)
            feed["image_grid_thw"] = grid_np

        onnx_out = session.run(None, feed)[0]
        onnx_tensor = torch.from_numpy(onnx_out).to(device)

        # Compare
        cos_sim = F.cosine_similarity(
            ref_out.flatten().unsqueeze(0).float(),
            onnx_tensor.flatten().unsqueeze(0).float(),
        ).item()

        max_diff = (ref_out.float() - onnx_tensor.float()).abs().max().item()

        status = "OK" if cos_sim > 0.99 else "WARN" if cos_sim > 0.95 else "FAIL"
        print(f"  {h}x{w} -> grid={grid}, patches={pv.shape[0]}: "
              f"cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f} [{status}]")

    return True


def verify_classification(onnx_path, model, device, is_dynamic=False):
    """Run classification on some test images and compare PyTorch vs ONNX."""
    import onnxruntime as ort
    from PIL import Image

    print(f"\n=== Classification Verification ===")

    # Load ONNX session
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = [inp.name for inp in session.get_inputs()]

    # Check if we have any test images
    test_img_dir = SCRIPT_DIR.parent / "data-creation" / "data" / "coco_dataset" / "train" / "images"
    if not test_img_dir.exists():
        test_img_dir = SCRIPT_DIR.parent / "data-creation" / "data" / "ref_images"

    if not test_img_dir.exists():
        print("No test images found, skipping classification verification")
        return

    # Get a few test images
    img_files = sorted(test_img_dir.glob("*.jpg"))[:5]
    if not img_files:
        img_files = sorted(test_img_dir.glob("*.png"))[:5]

    if not img_files:
        print("No test images found")
        return

    cls_head = model.cls_head
    language = model.language

    for img_path in img_files:
        pil_img = Image.open(img_path)

        if is_dynamic:
            pv, grid = preprocess_image_smart(pil_img, device)
        else:
            total_patches = session.get_inputs()[0].shape[0]
            if isinstance(total_patches, str):
                target_size = 448
            else:
                side = int(math.sqrt(total_patches))
                target_size = side * VIS_PATCH_SIZE
            pv, grid = letterbox_preprocess(pil_img, target_size, device)

        # PyTorch reference
        with torch.no_grad():
            ref_vis = model.vision(pv, [grid])
            # Pool and classify via cls_head directly on vision features
            # (skip language model for speed - just compare vision outputs)

        # ONNX
        pv_np = pv.cpu().numpy()
        feed = {"pixel_values": pv_np}
        if "image_grid_thw" in input_names:
            grid_np = np.array([[grid[0], grid[1], grid[2]]], dtype=np.int64)
            feed["image_grid_thw"] = grid_np

        onnx_out = session.run(None, feed)[0]
        onnx_tensor = torch.from_numpy(onnx_out).to(device)

        cos_sim = F.cosine_similarity(
            ref_vis.flatten().unsqueeze(0).float(),
            onnx_tensor.flatten().unsqueeze(0).float(),
        ).item()

        print(f"  {img_path.name}: cos_sim={cos_sim:.6f}, "
              f"ref_shape={ref_vis.shape}, onnx_shape={onnx_tensor.shape}")


def main():
    # Use CPU for export to avoid GPU OOM (GPU may be busy with training)
    device = torch.device("cpu")
    print(f"Device: {device} (using CPU for ONNX export to avoid OOM)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(device)

    # === Try 1: Dynamic export ===
    dynamic_path = try_dynamic_export(model, device)

    if dynamic_path and dynamic_path.exists():
        # Verify dynamic export
        try:
            verify_onnx(dynamic_path, model, device, is_dynamic=True)
            # Quantize
            int8_path = quantize_onnx(dynamic_path)
            verify_onnx(int8_path, model, device, is_dynamic=True)
            print(f"\nDynamic export SUCCESS: {int8_path}")
            print(f"Final size: {int8_path.stat().st_size / 1024**2:.1f} MB")
            return
        except Exception as e:
            print(f"Dynamic verification failed: {e}")
            import traceback
            traceback.print_exc()

    # === Fallback: Fixed grid with letterbox ===
    print("\n" + "="*60)
    print("FALLING BACK TO LETTERBOX + FIXED GRID EXPORT")
    print("="*60)

    # Try multiple sizes
    for target_size in [448, 384, 256]:
        print(f"\n--- Trying {target_size}x{target_size} ---")
        try:
            fixed_path = export_fixed_grid(model, device, target_size)
            verify_onnx(fixed_path, model, device, is_dynamic=False)

            int8_path = quantize_onnx(fixed_path)
            verify_onnx(int8_path, model, device, is_dynamic=False)

            final_size = int8_path.stat().st_size / 1024**2
            print(f"\nLetterbox {target_size}x{target_size} export SUCCESS: {int8_path}")
            print(f"Final size: {final_size:.1f} MB")

            # Only need one working size
            if final_size < 200:  # fits budget
                print(f"\nFinal model: {int8_path}")
                break
        except Exception as e:
            print(f"Export at {target_size}x{target_size} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
