"""
run.py -- Hybrid YOLO + DINOv2 inference pipeline for NM i AI 2026 Object Detection.

Usage: python run.py --data /data/images/ --output /output.json

Two-stage pipeline:
  1. YOLO detects product bounding boxes (+ first-pass classification)
  2. DINOv2 embeds each crop and matches against reference product embeddings
  3. Fusion: combine YOLO and DINOv2 predictions for final classification
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from ultralytics import YOLO


def load_dinov2(weights_path: Path, device: torch.device):
    """Load DINOv2 ViT-S/14 model."""
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)

    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    model.eval().to(device)

    data_config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_config, is_training=False)

    return model, transform


def load_ref_embeddings(path: Path, device: torch.device):
    """Load pre-computed reference embeddings.

    Returns:
        ref_embeddings: dict {category_id: tensor [N_angles, embed_dim]}
        ref_matrix: tensor [total_refs, embed_dim] flattened
        ref_cat_ids: tensor [total_refs] category ID per row in ref_matrix
    """
    data = torch.load(path, map_location=device, weights_only=False)

    if isinstance(data, dict) and "embeddings" in data:
        emb_dict = data["embeddings"]
    else:
        emb_dict = data

    ref_embeddings = {}
    ref_list = []
    cat_id_list = []

    for cat_id, emb in emb_dict.items():
        cat_id = int(cat_id)
        if isinstance(emb, torch.Tensor):
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            emb = F.normalize(emb.float().to(device), dim=-1)
            ref_embeddings[cat_id] = emb
            ref_list.append(emb)
            cat_id_list.extend([cat_id] * emb.shape[0])

    ref_matrix = torch.cat(ref_list, dim=0) if ref_list else None
    ref_cat_ids = torch.tensor(cat_id_list, device=device) if cat_id_list else None

    return ref_embeddings, ref_matrix, ref_cat_ids


@torch.no_grad()
def embed_crops(model, transform, crops: list, device: torch.device, batch_size: int = 64):
    """Extract DINOv2 embeddings for a list of PIL image crops."""
    if not crops:
        return torch.empty(0, model.num_features, device=device)

    all_embeddings = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        tensors = torch.stack([transform(c.convert("RGB")) for c in batch]).to(device)
        embs = model(tensors)
        embs = F.normalize(embs, dim=-1)
        all_embeddings.append(embs)

    return torch.cat(all_embeddings, dim=0)


def match_per_category(embeddings, ref_embeddings):
    """Match each query embedding against reference categories (max over angles)."""
    N = embeddings.shape[0]
    device = embeddings.device
    best_cats = torch.zeros(N, dtype=torch.long, device=device)
    best_sims = torch.full((N,), -1.0, device=device)

    for cat_id, ref_emb in ref_embeddings.items():
        sim = embeddings @ ref_emb.T  # [N, M]
        max_sim = sim.max(dim=-1).values  # [N]
        better = max_sim > best_sims
        best_sims[better] = max_sim[better]
        best_cats[better] = cat_id

    return best_cats, best_sims


def fuse_predictions(
    yolo_classes, yolo_confs, dino_classes, dino_sims,
    yolo_trust=0.7, dino_trust=0.5
):
    """Fuse YOLO classification with DINOv2 nearest-neighbor results."""
    final_classes = yolo_classes.clone()
    final_confs = yolo_confs.clone()

    yolo_confident = yolo_confs > yolo_trust
    dino_confident = dino_sims > dino_trust
    dino_agrees = dino_classes == yolo_classes

    # Both agree and confident -> boost confidence
    agreed = yolo_confident & dino_agrees
    final_confs[agreed] = torch.max(yolo_confs[agreed], dino_sims[agreed])

    # YOLO uncertain, DINOv2 confident -> override
    override = ~yolo_confident & dino_confident
    final_classes[override] = dino_classes[override]
    final_confs[override] = dino_sims[override]

    # Both confident but disagree -> trust higher confidence
    disagree = yolo_confident & dino_confident & ~dino_agrees
    dino_wins = disagree & (dino_sims > yolo_confs)
    final_classes[dino_wins] = dino_classes[dino_wins]
    final_confs[dino_wins] = dino_sims[dino_wins]

    return final_classes, final_confs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = Path(__file__).parent

    # --- Load models ---
    # YOLO detector
    yolo_path = script_dir / "model.pt"
    if not yolo_path.exists():
        # Try alternative names
        for name in ["best.pt", "yolo.pt", "yolo11x.pt"]:
            alt = script_dir / name
            if alt.exists():
                yolo_path = alt
                break

    yolo = YOLO(str(yolo_path))

    # DINOv2 classifier
    dinov2_weights = script_dir / "dinov2_vits14.pth"
    dino_model, dino_transform = load_dinov2(dinov2_weights, device)

    # Reference embeddings
    ref_path = script_dir / "ref_embeddings.pth"
    has_ref = ref_path.exists()
    ref_embeddings = ref_matrix = ref_cat_ids = None
    if has_ref:
        ref_embeddings, ref_matrix, ref_cat_ids = load_ref_embeddings(ref_path, device)

    # Optional: linear probe
    probe_path = script_dir / "linear_probe.pth"
    linear_probe = None
    if probe_path.exists():
        linear_probe = torch.load(probe_path, map_location=device, weights_only=False)
        if isinstance(linear_probe, dict):
            first_key = next(iter(linear_probe))
            weight = linear_probe[first_key]
            num_classes, embed_dim = weight.shape
            probe = torch.nn.Linear(embed_dim, num_classes)
            probe.load_state_dict(linear_probe)
            linear_probe = probe.eval().to(device)

    # --- Process images ---
    images_dir = Path(args.data)
    image_paths = sorted(images_dir.glob("*.jpg"))

    results = []

    for img_path in image_paths:
        # Stage 1: YOLO detection
        dets = yolo.predict(
            str(img_path),
            conf=0.001,
            iou=0.4,
            imgsz=1024,
            verbose=False,
            device=device,
        )

        boxes = dets[0].boxes
        if boxes is None or len(boxes) == 0:
            results.append({"image_id": img_path.name, "predictions": []})
            continue

        # Extract YOLO predictions
        xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]
        xywh = boxes.xywh.cpu().numpy()  # [N, 4] center format
        yolo_classes = boxes.cls.long()   # [N]
        yolo_confs = boxes.conf           # [N]

        # Convert to COCO format [x_min, y_min, width, height]
        bboxes_coco = np.column_stack([
            xyxy[:, 0],              # x_min
            xyxy[:, 1],              # y_min
            xyxy[:, 2] - xyxy[:, 0], # width
            xyxy[:, 3] - xyxy[:, 1], # height
        ])

        # Stage 2: DINOv2 classification (if reference embeddings available)
        if has_ref and ref_embeddings:
            img = Image.open(img_path).convert("RGB")
            crops = []
            for x1, y1, x2, y2 in xyxy:
                # Clamp to image bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img.width, int(x2))
                y2 = min(img.height, int(y2))
                if x2 > x1 and y2 > y1:
                    crops.append(img.crop((x1, y1, x2, y2)))
                else:
                    # Degenerate box — use tiny placeholder
                    crops.append(Image.new("RGB", (32, 32), (128, 128, 128)))

            embeddings = embed_crops(dino_model, dino_transform, crops, device)

            # Match against references
            dino_classes, dino_sims = match_per_category(embeddings, ref_embeddings)

            # Fuse
            final_classes, final_confs = fuse_predictions(
                yolo_classes.to(device), yolo_confs.to(device),
                dino_classes, dino_sims,
            )
            final_classes = final_classes.cpu().numpy()
            final_confs = final_confs.cpu().numpy()
        else:
            # No DINOv2 — use YOLO only
            final_classes = yolo_classes.cpu().numpy()
            final_confs = yolo_confs.cpu().numpy()

        # Build predictions
        predictions = []
        for i in range(len(xyxy)):
            predictions.append({
                "bbox": [
                    float(bboxes_coco[i, 0]),
                    float(bboxes_coco[i, 1]),
                    float(bboxes_coco[i, 2]),
                    float(bboxes_coco[i, 3]),
                ],
                "category_id": int(final_classes[i]),
                "confidence": float(final_confs[i]),
            })

        results.append({"image_id": img_path.name, "predictions": predictions})

    # Write output
    with open(args.output, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
