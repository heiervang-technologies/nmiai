#!/usr/bin/env python3
"""Recursive convolutional model for Astar Island cellular automaton prediction.

Architecture:
- Encode 40x40 grid as one-hot + spatial features -> multi-channel input
- Apply a small learned convolutional block T times (shared weights = recurrent)
- Each step: conv -> batch norm -> relu -> conv -> softmax blend with previous state
- Output: 40x40x6 probability distribution per cell
- Train by minimizing KL divergence against ground truth

The model learns approximate transition rules from neighborhood context,
applied iteratively to propagate information across the grid.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
MODEL_PATH = BASE_DIR / "recursive_model.pt"

N_CLASSES = 6
GRID_SIZE = 40

# Cell code -> class index mapping
# Initial codes: 0/10/11 -> class 0 (Empty/Ocean/Plains), 1->1, 2->2, 4->4, 5->5
# But we need to encode the initial type distinctly for the model
CELL_CODES = [0, 1, 2, 4, 5, 10, 11]
N_CELL_TYPES = len(CELL_CODES)  # 7 distinct initial types
CODE_TO_IDX = {code: i for i, code in enumerate(CELL_CODES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_grid(initial_grid):
    """Encode initial grid into model input tensor.

    Returns: (C, H, W) tensor with one-hot cell types + spatial features
    """
    ig = np.array(initial_grid, dtype=np.int32)
    H, W = ig.shape

    # One-hot encode cell types (7 channels)
    one_hot = np.zeros((N_CELL_TYPES, H, W), dtype=np.float32)
    for code, idx in CODE_TO_IDX.items():
        one_hot[idx] = (ig == code).astype(np.float32)

    # Static mask channels: ocean (never changes) and mountain (never changes)
    ocean_mask = (ig == 10).astype(np.float32)[np.newaxis]
    mountain_mask = (ig == 5).astype(np.float32)[np.newaxis]

    # Normalized position features (help model learn spatial patterns)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
    pos = np.stack([yy, xx], axis=0).astype(np.float32)

    # Distance to nearest settlement/port (normalized)
    from scipy.ndimage import distance_transform_edt
    civ_mask = (ig == 1) | (ig == 2)
    if civ_mask.any():
        dist_civ = distance_transform_edt(~civ_mask).astype(np.float32)
        dist_civ = dist_civ / (dist_civ.max() + 1e-6)
    else:
        dist_civ = np.ones((H, W), dtype=np.float32)

    # Distance to ocean
    ocean_bool = ig == 10
    if ocean_bool.any():
        dist_ocean = distance_transform_edt(~ocean_bool).astype(np.float32)
        dist_ocean = dist_ocean / (dist_ocean.max() + 1e-6)
    else:
        dist_ocean = np.ones((H, W), dtype=np.float32)

    features = np.concatenate([
        one_hot,           # 7 channels
        ocean_mask,        # 1 channel
        mountain_mask,     # 1 channel
        pos,               # 2 channels
        dist_civ[np.newaxis],    # 1 channel
        dist_ocean[np.newaxis],  # 1 channel
    ], axis=0)  # Total: 13 channels

    return features


def initial_state_from_grid(initial_grid):
    """Create initial probability state from grid (deterministic initial assignment).

    Returns: (6, H, W) tensor of initial class probabilities
    """
    ig = np.array(initial_grid, dtype=np.int32)
    H, W = ig.shape

    # Map initial codes to class probabilities
    # Class mapping: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
    state = np.zeros((N_CLASSES, H, W), dtype=np.float32)

    # Ocean (10) and Plains (11) -> start as class 0 (Empty)
    state[0] = ((ig == 10) | (ig == 11) | (ig == 0)).astype(np.float32)
    state[1] = (ig == 1).astype(np.float32)  # Settlement
    state[2] = (ig == 2).astype(np.float32)  # Port
    # Class 3 = Ruin (not in initial grid)
    state[4] = (ig == 4).astype(np.float32)  # Forest
    state[5] = (ig == 5).astype(np.float32)  # Mountain

    return state


class RecursiveCABlock(nn.Module):
    """A single convolutional transition block applied recursively.

    Takes current state (6 channels) + context features (13 channels) = 19 channels
    Outputs updated state probabilities (6 channels)
    """

    def __init__(self, n_context=13, n_classes=6, hidden=48):
        super().__init__()
        in_ch = n_classes + n_context  # 19

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, n_classes, 1),  # 1x1 to produce class logits
        )

        # Learnable gate: how much to update vs keep previous state
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, state, context):
        """
        state: (B, 6, H, W) - current probability state
        context: (B, 13, H, W) - static context features
        Returns: (B, 6, H, W) - updated probability state
        """
        x = torch.cat([state, context], dim=1)
        new_logits = self.net(x)
        new_state = F.softmax(new_logits, dim=1)

        gate = self.gate(x)  # (B, 1, H, W)
        # Blend: gate controls how much to move toward new state
        updated = gate * new_state + (1 - gate) * state
        # Re-normalize to ensure valid probabilities
        updated = updated / (updated.sum(dim=1, keepdim=True) + 1e-8)
        return updated


class RecursiveCAModel(nn.Module):
    """Recursive cellular automaton model.

    Applies a shared transition block T times to evolve initial state
    into predicted final probability distribution.
    """

    def __init__(self, n_context=13, n_classes=6, hidden=48, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        self.n_classes = n_classes

        # Shared transition block (same weights at each step)
        self.transition = RecursiveCABlock(n_context, n_classes, hidden)

        # Static cell masks are handled in forward pass

    def forward(self, context, initial_state, static_mask):
        """
        context: (B, 13, H, W) - encoded grid features
        initial_state: (B, 6, H, W) - initial class probabilities
        static_mask: (B, 1, H, W) - 1 for static cells (ocean/mountain), 0 for dynamic

        Returns: (B, 6, H, W) - final predicted probabilities
        """
        state = initial_state

        for step in range(self.n_steps):
            new_state = self.transition(state, context)
            # Preserve static cells
            state = static_mask * initial_state + (1 - static_mask) * new_state

        return state


def load_data():
    """Load all ground truth data."""
    data = []
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        parts = f.stem.split("_")
        round_num = int(parts[0].replace("round", ""))
        seed_num = int(parts[1].replace("seed", ""))

        with open(f) as fh:
            d = json.load(fh)

        if "ground_truth" not in d or "initial_grid" not in d:
            continue

        data.append({
            "round": round_num,
            "seed": seed_num,
            "initial_grid": d["initial_grid"],
            "ground_truth": np.array(d["ground_truth"], dtype=np.float32),
        })
    return data


def prepare_sample(sample):
    """Convert a single sample to tensors."""
    ig = sample["initial_grid"]

    context = encode_grid(ig)  # (13, H, W)
    initial_state = initial_state_from_grid(ig)  # (6, H, W)
    gt = sample["ground_truth"].transpose(2, 0, 1)  # (6, H, W)

    ig_arr = np.array(ig, dtype=np.int32)
    static = ((ig_arr == 10) | (ig_arr == 5)).astype(np.float32)[np.newaxis]  # (1, H, W)

    return (
        torch.tensor(context),
        torch.tensor(initial_state),
        torch.tensor(static),
        torch.tensor(gt),
    )


def kl_loss(pred, target):
    """Entropy-weighted KL divergence loss: KL(target || pred).

    This matches the competition scoring metric.
    """
    # target = p (ground truth), pred = q (our prediction)
    p = target.clamp(min=1e-8)
    q = pred.clamp(min=1e-8)

    # Per-cell KL: sum over classes
    kl = (p * (p.log() - q.log())).sum(dim=1)  # (B, H, W)

    # Entropy weighting: high-entropy cells matter more
    entropy = -(p * p.log()).sum(dim=1)  # (B, H, W)

    # Weight by entropy (cells with entropy ~0 are static and trivial)
    weight = 0.25 + entropy
    weighted_kl = weight * kl

    return weighted_kl.mean()


def train_model(n_epochs=600, lr=1e-3, n_steps=8, hidden=48, holdout_round=None):
    """Train the recursive CA model.

    Args:
        holdout_round: if set, exclude this round from training (for CV)
    """
    all_data = load_data()

    if holdout_round is not None:
        train_data = [s for s in all_data if s["round"] != holdout_round]
        val_data = [s for s in all_data if s["round"] == holdout_round]
    else:
        train_data = all_data
        val_data = []

    log.info(f"Training on {len(train_data)} samples, validating on {len(val_data)}")

    # Prepare all training tensors
    train_tensors = [prepare_sample(s) for s in train_data]
    val_tensors = [prepare_sample(s) for s in val_data]

    model = RecursiveCAModel(n_context=13, n_classes=N_CLASSES, hidden=hidden, n_steps=n_steps)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        # Shuffle training data
        indices = np.random.permutation(len(train_tensors))

        for idx in indices:
            context, init_state, static, gt = [t.unsqueeze(0).to(DEVICE) for t in train_tensors[idx]]

            pred = model(context, init_state, static)
            loss = kl_loss(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_tensors)

        # Validate
        if val_tensors and (epoch + 1) % 20 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for context, init_state, static, gt in val_tensors:
                    context, init_state, static, gt = [t.unsqueeze(0).to(DEVICE) for t in (context, init_state, static, gt)]
                    pred = model(context, init_state, static)
                    val_loss += kl_loss(pred, gt).item()
            avg_val_loss = val_loss / len(val_tensors)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            log.info(f"Epoch {epoch+1}/{n_epochs}: train={avg_train_loss:.6f} val={avg_val_loss:.6f} best_val={best_val_loss:.6f}")
        elif (epoch + 1) % 50 == 0:
            log.info(f"Epoch {epoch+1}/{n_epochs}: train={avg_train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_and_save():
    """Train on all data and save model weights."""
    model = train_model(n_epochs=800, lr=1e-3, n_steps=8, hidden=48)
    model = model.cpu()
    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"Model saved to {MODEL_PATH}")
    return model


# Global model cache for predict()
_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not MODEL_PATH.exists():
        log.info("No saved model found, training from scratch...")
        _model_cache = train_and_save()
    else:
        _model_cache = RecursiveCAModel(n_context=13, n_classes=N_CLASSES, hidden=48, n_steps=8)
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        _model_cache.load_state_dict(state)
        log.info(f"Loaded model from {MODEL_PATH}")

    _model_cache.eval()
    return _model_cache


def predict(initial_grid):
    """Predict 40x40x6 probability tensor from initial grid.

    Args:
        initial_grid: 40x40 list/array of cell codes

    Returns:
        40x40x6 numpy array of class probabilities
    """
    model = _load_model()

    context = torch.tensor(encode_grid(initial_grid)).unsqueeze(0)  # (1, 13, H, W)
    init_state = torch.tensor(initial_state_from_grid(initial_grid)).unsqueeze(0)  # (1, 6, H, W)

    ig_arr = np.array(initial_grid, dtype=np.int32)
    static = torch.tensor(
        ((ig_arr == 10) | (ig_arr == 5)).astype(np.float32)[np.newaxis, np.newaxis]
    )  # (1, 1, H, W)

    with torch.no_grad():
        pred = model(context, init_state, static)  # (1, 6, H, W)

    # Convert to numpy (H, W, 6)
    result = pred[0].permute(1, 2, 0).numpy()

    # Floor at 0.01 and renormalize
    result = np.maximum(result, 0.01)
    result /= result.sum(axis=2, keepdims=True)

    return result


if __name__ == "__main__":
    if "--train" in sys.argv:
        train_and_save()
    elif "--cv" in sys.argv:
        # Cross-validation
        all_data = load_data()
        rounds = sorted(set(s["round"] for s in all_data))
        log.info(f"Running leave-one-round-out CV over {len(rounds)} rounds")

        cv_scores = []
        for holdout in rounds:
            model = train_model(n_epochs=400, lr=1e-3, n_steps=8, hidden=48, holdout_round=holdout)
            model = model.cpu().eval()

            holdout_data = [s for s in all_data if s["round"] == holdout]
            losses = []
            for sample in holdout_data:
                pred_np = np.zeros((GRID_SIZE, GRID_SIZE, N_CLASSES), dtype=np.float32)
                context = torch.tensor(encode_grid(sample["initial_grid"])).unsqueeze(0)
                init_state = torch.tensor(initial_state_from_grid(sample["initial_grid"])).unsqueeze(0)
                ig_arr = np.array(sample["initial_grid"], dtype=np.int32)
                static = torch.tensor(
                    ((ig_arr == 10) | (ig_arr == 5)).astype(np.float32)[np.newaxis, np.newaxis]
                )
                with torch.no_grad():
                    pred = model(context, init_state, static)
                pred_np = pred[0].permute(1, 2, 0).numpy()
                pred_np = np.maximum(pred_np, 0.01)
                pred_np /= pred_np.sum(axis=2, keepdims=True)

                gt = sample["ground_truth"]
                # Compute weighted KL
                p = np.maximum(gt, 1e-10)
                q = np.maximum(pred_np, 1e-10)
                kl = np.sum(p * np.log(p / q), axis=2)
                H = -np.sum(p * np.log(p), axis=2)
                weight = 0.25 + H
                wkl = float(np.sum(weight * kl) / np.sum(weight))
                losses.append(wkl)

            avg = np.mean(losses)
            cv_scores.append(avg)
            log.info(f"Round {holdout} holdout: weighted KL = {avg:.6f}")

        log.info(f"CV mean weighted KL: {np.mean(cv_scores):.6f}")
    else:
        # Default: train and save
        train_and_save()
