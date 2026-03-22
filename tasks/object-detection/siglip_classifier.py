"""SigLIP classifier for submission integration.

Uses timm vision encoder + learned classification head.
No transformers library needed.
"""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from timm.data import create_transform, resolve_data_config

SCRIPT_DIR = Path(__file__).parent
NUM_CLASSES = 356


class SigLIPClassifier:
    """Ready-to-use SigLIP classifier for YOLO crop classification."""

    def __init__(self, model_dir, device=None, model_name="vit_so400m_patch14_siglip_384"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = Path(model_dir)

        # Load vision encoder
        # Try to load local weights first, then pretrained
        weights_path = model_dir / "siglip_vision.pth"
        if weights_path.exists():
            self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        else:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)

        self.model.eval().to(self.device)
        self.embed_dim = self.model.num_features

        # Load classification head
        head_path = model_dir / "siglip_head.pth"
        if head_path.exists():
            head_data = torch.load(head_path, map_location=self.device, weights_only=True)
            self.head_weight = head_data["head_weight"].to(self.device).float()
            self.logit_scale = head_data.get("logit_scale", torch.tensor(0.0)).to(self.device)
        else:
            # Fallback: use text embeddings as head
            text_path = model_dir / "text_embeddings_matrix.pth"
            self.head_weight = torch.load(text_path, map_location=self.device, weights_only=True).float()
            self.logit_scale = torch.tensor(0.0, device=self.device)

        # Normalize head weights
        self.head_weight_norm = F.normalize(self.head_weight, dim=-1)

        # Get transforms
        config = resolve_data_config(self.model.pretrained_cfg)
        self.transform = create_transform(**config, is_training=False)

    @torch.inference_mode()
    def classify_crops(self, pil_crops, batch_size=64):
        """Classify a list of PIL Image crops.

        Returns:
            category_ids: np.array of predicted category IDs
            confidences: np.array of confidence scores
        """
        all_cats = []
        all_confs = []

        for i in range(0, len(pil_crops), batch_size):
            batch_crops = pil_crops[i:i + batch_size]
            tensors = torch.stack([self.transform(crop) for crop in batch_crops]).to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                embeddings = self.model(tensors)

            embeddings = F.normalize(embeddings.float(), dim=-1)
            logits = embeddings @ self.head_weight_norm.T * self.logit_scale.exp()
            probs = F.softmax(logits, dim=-1)
            confs, cats = probs.max(dim=-1)

            all_cats.extend(cats.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

        return np.array(all_cats, dtype=np.int64), np.array(all_confs, dtype=np.float32)

    @torch.inference_mode()
    def classify_boxes(self, image_bgr, boxes, detector_labels, detector_scores,
                       det_weight=0.15, batch_size=64):
        """Classify boxes in an image, with optional detector prior blending.

        Compatible with the existing submission pipeline interface.
        """
        import cv2
        if len(boxes) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Extract crops
        crops = []
        for box in boxes:
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(image_pil.width, int(box[2])), min(image_pil.height, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(Image.new("RGB", (32, 32), (114, 114, 114)))
            else:
                crops.append(image_pil.crop((x1, y1, x2, y2)))

        category_ids, confidences = self.classify_crops(crops, batch_size)

        if det_weight > 0 and detector_labels is not None:
            # Blend with detector prior (same as current pipeline)
            # Re-compute with blending
            all_cats = []
            all_confs = []
            for i in range(0, len(crops), batch_size):
                batch_crops = crops[i:i + batch_size]
                batch_det_labels = detector_labels[i:i + batch_size]
                batch_det_scores = detector_scores[i:i + batch_size]

                tensors = torch.stack([self.transform(crop) for crop in batch_crops]).to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=self.device.type == "cuda"):
                    embeddings = self.model(tensors)
                embeddings = F.normalize(embeddings.float(), dim=-1)
                logits = embeddings @ self.head_weight_norm.T * self.logit_scale.exp()
                siglip_prob = F.softmax(logits, dim=-1)

                # Detector prior
                prior = torch.zeros((len(batch_crops), NUM_CLASSES), device=self.device)
                for j in range(len(batch_crops)):
                    prior[j, int(batch_det_labels[j])] = float(batch_det_scores[j])
                row_sum = prior.sum(dim=-1, keepdim=True)
                prior = prior / row_sum.clamp(min=1e-8)

                combined = siglip_prob * (1 - det_weight) + prior * det_weight
                confs, cats = combined.max(dim=-1)
                all_cats.extend(cats.cpu().tolist())
                all_confs.extend(confs.cpu().tolist())

            category_ids = np.array(all_cats, dtype=np.int64)
            confidences = np.array(all_confs, dtype=np.float32)

        return category_ids, confidences
