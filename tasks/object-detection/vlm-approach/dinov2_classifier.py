"""
DINOv2-based product classifier using reference image embeddings.

Loads pre-computed reference embeddings and classifies crop images by
nearest-neighbor cosine similarity matching.
"""

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from pathlib import Path


class DINOv2Classifier:
    """Classify product crops by matching DINOv2 embeddings to reference images."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        ref_embeddings_path: str | Path | None = None,
        linear_probe_path: str | Path | None = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Load DINOv2 model via timm
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.model.eval().to(self.device)

        # Build preprocessing transform from model config
        data_config = resolve_data_config(self.model.pretrained_cfg)
        self.transform = create_transform(**data_config, is_training=False)

        # Reference embeddings: {category_id: tensor [N_angles, embed_dim]}
        self.ref_embeddings = None
        self.ref_category_ids = None
        self.ref_matrix = None  # [total_refs, embed_dim] flattened for batch matching

        # Optional linear probe
        self.linear_probe = None

        if ref_embeddings_path is not None:
            self.load_ref_embeddings(ref_embeddings_path)

        if linear_probe_path is not None:
            self.load_linear_probe(linear_probe_path)

    def load_weights(self, weights_path: str | Path):
        """Load DINOv2 model weights from a file."""
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def load_ref_embeddings(self, path: str | Path):
        """Load pre-computed reference embeddings.

        Expected format: dict with keys:
            - 'embeddings': dict {category_id (int): tensor [N, embed_dim]}
            - OR a flat dict {category_id (int): tensor [N, embed_dim]}
        """
        data = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(data, dict) and "embeddings" in data:
            embeddings_dict = data["embeddings"]
        else:
            embeddings_dict = data

        self.ref_embeddings = {}
        ref_list = []
        cat_id_list = []

        for cat_id, emb in embeddings_dict.items():
            cat_id = int(cat_id)
            if isinstance(emb, torch.Tensor):
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                emb = F.normalize(emb.float().to(self.device), dim=-1)
                self.ref_embeddings[cat_id] = emb
                ref_list.append(emb)
                cat_id_list.extend([cat_id] * emb.shape[0])

        if ref_list:
            self.ref_matrix = torch.cat(ref_list, dim=0)  # [total_refs, embed_dim]
            self.ref_category_ids = torch.tensor(cat_id_list, device=self.device)

    def load_linear_probe(self, path: str | Path):
        """Load a trained linear classifier head."""
        self.linear_probe = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(self.linear_probe, dict):
            # Assume it's a state dict for nn.Linear
            embed_dim = list(self.linear_probe.values())[0].shape[-1]
            num_classes = list(self.linear_probe.values())[0].shape[0]
            probe = torch.nn.Linear(embed_dim, num_classes)
            probe.load_state_dict(self.linear_probe)
            self.linear_probe = probe
        self.linear_probe.eval().to(self.device)

    @torch.no_grad()
    def embed_crops(self, crops: list[Image.Image], batch_size: int = 64) -> torch.Tensor:
        """Extract DINOv2 embeddings for a list of PIL image crops.

        Args:
            crops: List of PIL images (product crops from detection boxes).
            batch_size: Batch size for forward passes.

        Returns:
            Tensor of shape [N, embed_dim] with L2-normalized embeddings.
        """
        if not crops:
            return torch.empty(0, self.model.num_features, device=self.device)

        all_embeddings = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            batch_tensors = torch.stack(
                [self.transform(crop.convert("RGB")) for crop in batch_crops]
            ).to(self.device)

            embeddings = self.model(batch_tensors)  # [B, embed_dim]
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def match_nearest(
        self, embeddings: torch.Tensor, top_k: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match embeddings against reference database using cosine similarity.

        Args:
            embeddings: [N, embed_dim] L2-normalized query embeddings.
            top_k: Number of top matches to return per query.

        Returns:
            category_ids: [N, top_k] matched category IDs.
            similarities: [N, top_k] cosine similarity scores.
        """
        if self.ref_matrix is None:
            raise ValueError("No reference embeddings loaded. Call load_ref_embeddings first.")

        # Cosine similarity (embeddings are already normalized)
        sim = embeddings @ self.ref_matrix.T  # [N, total_refs]

        # Get top-k matches
        top_sims, top_indices = sim.topk(top_k, dim=-1)
        top_cat_ids = self.ref_category_ids[top_indices]

        return top_cat_ids, top_sims

    def match_nearest_per_category(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match embeddings, averaging similarity per category (handles multi-angle refs).

        For each query, computes max similarity to each category's reference images,
        then returns the best-matching category.

        Args:
            embeddings: [N, embed_dim] L2-normalized query embeddings.

        Returns:
            category_ids: [N] best matching category ID per query.
            similarities: [N] best similarity score per query.
        """
        if self.ref_embeddings is None:
            raise ValueError("No reference embeddings loaded.")

        N = embeddings.shape[0]
        best_cats = torch.zeros(N, dtype=torch.long, device=self.device)
        best_sims = torch.full((N,), -1.0, device=self.device)

        for cat_id, ref_emb in self.ref_embeddings.items():
            # ref_emb: [M, embed_dim] where M is number of reference angles
            # sim: [N, M]
            sim = embeddings @ ref_emb.T
            # Take max across angles for each query
            max_sim = sim.max(dim=-1).values  # [N]

            better = max_sim > best_sims
            best_sims[better] = max_sim[better]
            best_cats[better] = cat_id

        return best_cats, best_sims

    def classify_with_probe(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Classify using the trained linear probe.

        Args:
            embeddings: [N, embed_dim] embeddings (normalization not required).

        Returns:
            category_ids: [N] predicted category IDs.
            confidences: [N] softmax confidence scores.
        """
        if self.linear_probe is None:
            raise ValueError("No linear probe loaded.")

        logits = self.linear_probe(embeddings)
        probs = F.softmax(logits, dim=-1)
        confidences, category_ids = probs.max(dim=-1)

        return category_ids, confidences

    def classify(
        self,
        crops: list[Image.Image],
        yolo_classes: torch.Tensor | None = None,
        yolo_confs: torch.Tensor | None = None,
        batch_size: int = 64,
        yolo_trust_threshold: float = 0.7,
        dino_trust_threshold: float = 0.5,
        use_probe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full classification pipeline: embed crops and match/classify.

        Fusion logic:
        1. If linear probe is available and use_probe=True, use it
        2. If YOLO is confident AND DINOv2 agrees, trust YOLO
        3. If DINOv2 similarity is high, trust DINOv2
        4. Otherwise fall back to YOLO

        Args:
            crops: List of PIL images.
            yolo_classes: [N] YOLO predicted class IDs (optional).
            yolo_confs: [N] YOLO confidence scores (optional).
            batch_size: Batch size for embedding.
            yolo_trust_threshold: Trust YOLO above this confidence.
            dino_trust_threshold: Trust DINOv2 above this similarity.
            use_probe: Whether to use linear probe instead of nearest-neighbor.

        Returns:
            final_classes: [N] final predicted category IDs.
            final_confs: [N] final confidence scores.
        """
        if not crops:
            return torch.empty(0, dtype=torch.long), torch.empty(0)

        embeddings = self.embed_crops(crops, batch_size)
        N = len(crops)

        # Get DINOv2 predictions
        if use_probe and self.linear_probe is not None:
            dino_cats, dino_confs = self.classify_with_probe(embeddings)
        elif self.ref_matrix is not None:
            dino_cats, dino_confs = self.match_nearest_per_category(embeddings)
        else:
            # No reference embeddings and no probe — return YOLO predictions
            if yolo_classes is not None:
                return yolo_classes, yolo_confs if yolo_confs is not None else torch.ones(N)
            return torch.zeros(N, dtype=torch.long), torch.zeros(N)

        # If no YOLO predictions provided, return pure DINOv2
        if yolo_classes is None:
            return dino_cats, dino_confs

        # Fusion
        yolo_classes = yolo_classes.to(self.device)
        yolo_confs = yolo_confs.to(self.device) if yolo_confs is not None else torch.ones(N, device=self.device)

        final_classes = yolo_classes.clone()
        final_confs = yolo_confs.clone()

        # Where YOLO is confident AND DINOv2 agrees -> keep YOLO (high confidence)
        yolo_confident = yolo_confs > yolo_trust_threshold
        dino_agrees = dino_cats == yolo_classes
        agreed = yolo_confident & dino_agrees
        final_confs[agreed] = torch.max(yolo_confs[agreed], dino_confs[agreed])

        # Where DINOv2 is confident but YOLO isn't -> override with DINOv2
        dino_confident = dino_confs > dino_trust_threshold
        yolo_uncertain = ~yolo_confident
        override = dino_confident & yolo_uncertain
        final_classes[override] = dino_cats[override]
        final_confs[override] = dino_confs[override]

        # Where DINOv2 is confident and disagrees with confident YOLO -> weighted decision
        disagree = yolo_confident & dino_confident & ~dino_agrees
        # Trust whichever has higher confidence
        dino_wins = disagree & (dino_confs > yolo_confs)
        final_classes[dino_wins] = dino_cats[dino_wins]
        final_confs[dino_wins] = dino_confs[dino_wins]

        return final_classes, final_confs
