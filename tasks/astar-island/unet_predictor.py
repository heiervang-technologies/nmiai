#!/usr/bin/env python3
"""U-Net predictor for Astar Island.

Trains a tiny U-Net to predict 40x40x6 probability tensors from initial grids.
This is a regression task minimizing KL divergence on dynamic cells.

Usage:
    # Train and save model:
    uv run python3 unet_predictor.py train

    # Benchmark:
    uv run python3 benchmark.py unet_predictor
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Feature engineering ──────────────────────────────────────────────────────

CELL_CODES = [1, 2, 4, 5, 10, 11]
CODE_TO_IDX = {c: i for i, c in enumerate(CELL_CODES)}
NUM_CODES = len(CELL_CODES)

# Cell type meanings (from competition docs):
# 1 = grassland, 2 = forest, 4 = civilization, 5 = deep water (ocean)
# 10 = mountain, 11 = shallow water

OCEAN_CODES = {5}       # deep water
MOUNTAIN_CODES = {10}   # mountain
CIV_CODES = {4}         # civilization


def grid_to_features(initial_grid):
    """Convert 40x40 grid to feature tensor (C, 40, 40)."""
    grid = np.array(initial_grid, dtype=np.int32)
    H, W = grid.shape

    # One-hot encoding for each cell code (6 channels)
    one_hot = np.zeros((NUM_CODES, H, W), dtype=np.float32)
    for code, idx in CODE_TO_IDX.items():
        one_hot[idx] = (grid == code).astype(np.float32)

    # Distance to nearest civilization cell (1 channel)
    civ_mask = (grid == 4)
    if civ_mask.any():
        from scipy.ndimage import distance_transform_edt
        dist_civ = distance_transform_edt(~civ_mask).astype(np.float32)
        dist_civ = dist_civ / (dist_civ.max() + 1e-8)  # normalize to [0,1]
    else:
        dist_civ = np.ones((H, W), dtype=np.float32)

    # Distance to nearest ocean cell (1 channel)
    ocean_mask = (grid == 5)
    if ocean_mask.any():
        from scipy.ndimage import distance_transform_edt
        dist_ocean = distance_transform_edt(~ocean_mask).astype(np.float32)
        dist_ocean = dist_ocean / (dist_ocean.max() + 1e-8)
    else:
        dist_ocean = np.ones((H, W), dtype=np.float32)

    # Ocean adjacency: count of ocean neighbors in 3x3 (1 channel)
    ocean_float = ocean_mask.astype(np.float32)
    ocean_adj = np.zeros((H, W), dtype=np.float32)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(ocean_float, di, axis=0), dj, axis=1)
            ocean_adj += shifted
    ocean_adj /= 8.0  # normalize to [0,1]

    # Mountain adjacency (1 channel)
    mtn_float = (grid == 10).astype(np.float32)
    mtn_adj = np.zeros((H, W), dtype=np.float32)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(mtn_float, di, axis=0), dj, axis=1)
            mtn_adj += shifted
    mtn_adj /= 8.0

    # Stack: 6 one-hot + dist_civ + dist_ocean + ocean_adj + mtn_adj = 10 channels
    features = np.concatenate([
        one_hot,
        dist_civ[np.newaxis],
        dist_ocean[np.newaxis],
        ocean_adj[np.newaxis],
        mtn_adj[np.newaxis],
    ], axis=0)

    return features


# ── Data loading ─────────────────────────────────────────────────────────────

GT_DIR = Path(__file__).parent / "ground_truth"
MODEL_PATH = Path(__file__).parent / "unet_model.pt"


def load_all_data():
    """Load all ground truth files."""
    samples = []
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        name = f.stem
        parts = name.split("_")
        round_num = int(parts[0].replace("round", ""))
        seed_num = int(parts[1].replace("seed", ""))

        with open(f) as fh:
            data = json.load(fh)

        if "ground_truth" not in data or "initial_grid" not in data:
            continue

        samples.append({
            "round": round_num,
            "seed": seed_num,
            "initial_grid": data["initial_grid"],
            "ground_truth": np.array(data["ground_truth"], dtype=np.float32),
        })
    return samples


# ── U-Net Architecture ───────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """Tiny U-Net: 2 down, 2 up, max 32 channels."""

    def __init__(self, in_channels=10, out_channels=6):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.bottleneck = DoubleConv(64, 128)

        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Pad to make divisible by 4 (40 -> 40, already div by 4? 40/4=10, yes)
        e1 = self.enc1(x)        # (B, 32, 40, 40)
        e2 = self.enc2(self.pool(e1))  # (B, 64, 20, 20)
        b = self.bottleneck(self.pool(e2))  # (B, 128, 10, 10)

        d2 = self.up2(b)          # (B, 64, 20, 20)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 64, 20, 20)

        d1 = self.up1(d2)         # (B, 32, 40, 40)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, 32, 40, 40)

        out = self.final(d1)      # (B, 6, 40, 40)
        out = F.softmax(out, dim=1)
        return out


# ── Data augmentation ────────────────────────────────────────────────────────

def augment_batch(features, targets):
    """Apply random rotation and flips (the automaton is isotropic).

    features: (C, H, W) numpy
    targets: (H, W, 6) numpy
    Returns augmented copies.
    """
    # Random rotation: 0, 90, 180, 270
    k = np.random.randint(4)
    if k > 0:
        features = np.rot90(features, k, axes=(1, 2)).copy()
        targets = np.rot90(targets, k, axes=(0, 1)).copy()

    # Random horizontal flip
    if np.random.random() > 0.5:
        features = np.flip(features, axis=2).copy()
        targets = np.flip(targets, axis=1).copy()

    # Random vertical flip
    if np.random.random() > 0.5:
        features = np.flip(features, axis=1).copy()
        targets = np.flip(targets, axis=0).copy()

    return features, targets


# ── Training ─────────────────────────────────────────────────────────────────

def compute_kl_loss(pred, gt, mask):
    """KL divergence loss on dynamic cells only.

    pred: (B, 6, H, W) - predicted probabilities (after softmax)
    gt: (B, 6, H, W) - ground truth probabilities
    mask: (B, H, W) - True for dynamic cells
    """
    pred_safe = torch.clamp(pred, min=1e-6)
    gt_safe = torch.clamp(gt, min=1e-10)

    # KL(gt || pred) = sum(gt * log(gt/pred))
    kl_per_cell = (gt_safe * torch.log(gt_safe / pred_safe)).sum(dim=1)  # (B, H, W)

    # Entropy weighting
    entropy = -(gt_safe * torch.log2(gt_safe)).sum(dim=1)  # (B, H, W)
    weighted_kl = entropy * kl_per_cell

    # Mask and average
    if mask.any():
        return weighted_kl[mask].mean()
    return torch.tensor(0.0, device=pred.device)


def train_model(samples, val_rounds=None, epochs=2000, lr=1e-3, device='cuda', verbose=True):
    """Train U-Net on given samples.

    Args:
        samples: list of dicts with initial_grid, ground_truth, round, seed
        val_rounds: set of round numbers to hold out for validation
        epochs: max epochs
        lr: learning rate
        device: cuda or cpu
        verbose: print progress

    Returns:
        trained model, best validation loss (or final training loss)
    """
    if val_rounds is None:
        val_rounds = set()

    train_samples = [s for s in samples if s['round'] not in val_rounds]
    val_samples = [s for s in samples if s['round'] in val_rounds]

    # Precompute features
    train_features = []
    train_targets = []
    train_masks = []

    for s in train_samples:
        feat = grid_to_features(s['initial_grid'])
        gt = s['ground_truth']
        # Compute dynamic mask
        ent = -np.sum(np.maximum(gt, 1e-10) * np.log2(np.maximum(gt, 1e-10)), axis=2)
        mask = ent > 0.01
        train_features.append(feat)
        train_targets.append(gt)
        train_masks.append(mask)

    val_features = []
    val_targets = []
    val_masks = []
    for s in val_samples:
        feat = grid_to_features(s['initial_grid'])
        gt = s['ground_truth']
        ent = -np.sum(np.maximum(gt, 1e-10) * np.log2(np.maximum(gt, 1e-10)), axis=2)
        mask = ent > 0.01
        val_features.append(feat)
        val_targets.append(gt)
        val_masks.append(mask)

    model = TinyUNet(in_channels=10, out_channels=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_state = None
    patience = 300
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Shuffle and iterate through samples with augmentation
        indices = np.random.permutation(len(train_features))

        # Mini-batch: process all samples with augmentation
        batch_feats = []
        batch_targets = []
        batch_masks = []

        for i in indices:
            feat, tgt = augment_batch(train_features[i], train_targets[i])
            batch_feats.append(feat)
            batch_targets.append(tgt.transpose(2, 0, 1))  # (6, H, W)
            batch_masks.append(train_masks[i])

        feat_tensor = torch.tensor(np.stack(batch_feats), dtype=torch.float32, device=device)
        tgt_tensor = torch.tensor(np.stack(batch_targets), dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(np.stack(batch_masks), dtype=torch.bool, device=device)

        pred = model(feat_tensor)
        loss = compute_kl_loss(pred, tgt_tensor, mask_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss = loss.item()

        # Validation
        if val_samples and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vf = torch.tensor(np.stack(val_features), dtype=torch.float32, device=device)
                vt = torch.tensor(np.stack([t.transpose(2, 0, 1) for t in val_targets]), dtype=torch.float32, device=device)
                vm = torch.tensor(np.stack(val_masks), dtype=torch.bool, device=device)
                vpred = model(vf)
                val_loss = compute_kl_loss(vpred, vt, vm).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 10

            if verbose and (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}: train={total_loss:.6f} val={val_loss:.6f} best_val={best_val_loss:.6f}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        elif not val_samples and verbose and (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: train={total_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


# ── Prediction interface ─────────────────────────────────────────────────────

_model = None
_device = None


def _load_model():
    global _model, _device
    if _model is not None:
        return

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model = TinyUNet(in_channels=10, out_channels=6).to(_device)

    if MODEL_PATH.exists():
        state = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
        _model.load_state_dict(state)
        _model.eval()
    else:
        # No saved model - train on all data
        print("No saved model found, training from scratch...")
        samples = load_all_data()
        _model, _ = train_model(samples, epochs=2000, device=_device)
        torch.save(_model.state_dict(), MODEL_PATH)
        _model.eval()


def predict(initial_grid):
    """Predict 40x40x6 probability tensor from initial grid.

    Args:
        initial_grid: list of lists, 40x40, cell codes

    Returns:
        np.ndarray of shape (40, 40, 6) - probabilities
    """
    _load_model()

    features = grid_to_features(initial_grid)
    feat_tensor = torch.tensor(features[np.newaxis], dtype=torch.float32, device=_device)

    with torch.no_grad():
        pred = _model(feat_tensor)  # (1, 6, 40, 40)

    result = pred[0].cpu().numpy().transpose(1, 2, 0)  # (40, 40, 6)

    # Floor at 1e-6 and renormalize
    result = np.maximum(result, 1e-6)
    result /= result.sum(axis=2, keepdims=True)

    return result


# ── Cross-validation and training entrypoint ─────────────────────────────────

def run_cv():
    """Run leave-one-round-out cross-validation."""
    samples = load_all_data()
    rounds = sorted(set(s['round'] for s in samples))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Running leave-one-round-out CV on {len(rounds)} rounds, {len(samples)} samples")
    print(f"Device: {device}")

    cv_wkl = []

    for held_out in rounds:
        print(f"\n--- Holding out round {held_out} ---")
        model, val_loss = train_model(
            samples, val_rounds={held_out}, epochs=2000, device=device, verbose=True
        )

        # Evaluate on held-out round
        model.eval()
        held_samples = [s for s in samples if s['round'] == held_out]

        wkl_scores = []
        for s in held_samples:
            feat = grid_to_features(s['initial_grid'])
            feat_t = torch.tensor(feat[np.newaxis], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = model(feat_t)[0].cpu().numpy().transpose(1, 2, 0)

            pred = np.maximum(pred, 1e-6)
            pred /= pred.sum(axis=2, keepdims=True)

            gt = s['ground_truth']

            # Benchmark-style evaluation
            pred_eval = np.maximum(pred, 0.01)
            pred_eval /= pred_eval.sum(axis=2, keepdims=True)

            gt_safe = np.maximum(gt, 1e-10)
            pred_safe = np.maximum(pred_eval, 1e-10)
            kl = np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=2)
            H = -np.sum(gt_safe * np.log2(gt_safe), axis=2)
            wkl = H * kl
            dynamic = H > 0.01

            mean_wkl = wkl[dynamic].mean() if dynamic.any() else 0
            wkl_scores.append(mean_wkl)

        round_wkl = np.mean(wkl_scores)
        cv_wkl.append(round_wkl)
        print(f"  Round {held_out} wKL: {round_wkl:.6f}")

    print(f"\n=== CV Results ===")
    for r, wkl in zip(rounds, cv_wkl):
        print(f"  Round {r}: wKL = {wkl:.6f}")
    print(f"  Mean CV wKL: {np.mean(cv_wkl):.6f}")

    return cv_wkl


def train_final():
    """Train on all data and save model."""
    samples = load_all_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training final model on {len(samples)} samples")
    model, _ = train_model(samples, epochs=3000, device=device, verbose=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cv":
        run_cv()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        train_final()
    else:
        print("Usage: uv run python3 unet_predictor.py [cv|train]")
        print("  cv    - Run leave-one-round-out cross-validation")
        print("  train - Train final model on all data and save")
