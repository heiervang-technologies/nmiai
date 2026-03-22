#!/usr/bin/env python3
"""D3PM discrete diffusion for Astar Island - V2: Local feature conditioned.

V1 failed because it memorized grid layouts instead of learning dynamics.
V2 approach: model learns per-cell transition probabilities conditioned on
LOCAL FEATURES (same features BTP uses), not raw grid pixels.

For each cell, input features are:
- Cell type one-hot (6 classes)
- Distance to nearest settlement (bucketed)
- Number of ocean neighbors (0-8)
- Number of civ neighbors (0-8)
- Is coastal (bool)
- Position features (relative to settlements)

The diffusion part: we still use absorbing-state D3PM, but now the model
predicts per-cell class distributions given features. At inference, we sample
many times to build the probability tensor.

Key advantage over BTP: the neural net can learn NON-LINEAR interactions
between features that BTP's bucket lookup cannot capture.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import convolve, distance_transform_edt

BASE_DIR = Path(__file__).parent
REPLAY_DIR = BASE_DIR / "replays"
GT_DIR = BASE_DIR / "ground_truth"

N_CLASSES = 6
MASK_TOKEN = N_CLASSES
N_TOKENS = N_CLASSES + 1
H, W = 40, 40

CODE_TO_CLASS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4
PLAINS = 11

KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)


def code_to_class_grid(grid):
    out = np.zeros_like(grid, dtype=np.int64)
    for code, cls in CODE_TO_CLASS.items():
        out[grid == code] = cls
    return out


# ---------------------------------------------------------------------------
# Feature computation (matches BTP)
# ---------------------------------------------------------------------------

def compute_feature_maps(ig_raw):
    """Compute per-cell feature maps from initial grid.

    Returns dict of (H, W) feature arrays.
    """
    ig = ig_raw.astype(np.int32)
    h, w = ig.shape

    civ = (ig == SETTLEMENT) | (ig == PORT)
    ocean = ig == OCEAN
    mountain = ig == MOUNTAIN
    forest = ig == FOREST

    dist_civ = distance_transform_edt(~civ) if civ.any() else np.full((h, w), 99.0)
    n_ocean = convolve(ocean.astype(np.int32), KERNEL, mode="constant")
    n_civ = convolve(civ.astype(np.int32), KERNEL, mode="constant")
    coast = (n_ocean > 0) & ~ocean

    # Cell type one-hot for initial grid
    ig_classes = code_to_class_grid(ig)

    # Static mask
    static = ocean | mountain

    return {
        'ig_classes': ig_classes,          # (H, W) int
        'dist_civ': dist_civ.astype(np.float32),  # (H, W)
        'n_ocean': n_ocean.astype(np.float32),     # (H, W)
        'n_civ': n_civ.astype(np.float32),         # (H, W)
        'coast': coast.astype(np.float32),         # (H, W)
        'is_forest': forest.astype(np.float32),    # (H, W)
        'is_plains': (ig == PLAINS).astype(np.float32),
        'static': static,
    }


def features_to_tensor(feats):
    """Convert feature dict to (C, H, W) tensor.

    Channels:
    0-5: initial cell type one-hot (6)
    6: dist_civ (normalized)
    7: n_ocean / 8
    8: n_civ / 8
    9: coast
    10: is_forest
    11: is_plains
    = 12 channels
    """
    ig_onehot = np.eye(N_CLASSES, dtype=np.float32)[feats['ig_classes']]  # (H,W,6)

    stack = np.stack([
        np.clip(feats['dist_civ'] / 20.0, 0, 1),  # normalized dist
        feats['n_ocean'] / 8.0,
        feats['n_civ'] / 8.0,
        feats['coast'],
        feats['is_forest'],
        feats['is_plains'],
    ], axis=-1)  # (H, W, 6)

    combined = np.concatenate([ig_onehot, stack], axis=-1)  # (H, W, 12)
    return combined.transpose(2, 0, 1).astype(np.float32)   # (12, H, W)


N_FEAT_CH = 12  # feature channels


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def precompute_dataset():
    """Load replays, compute features, pack into GPU tensors."""
    feats_list = []
    finals_list = []
    statics_list = []

    for f in sorted(REPLAY_DIR.glob("round*_seed*.json")):
        if "dense_training" in str(f):
            continue
        data = json.loads(f.read_text())
        frames = data["frames"]
        ig_raw = np.array(frames[0]["grid"], dtype=np.int32)
        fg_raw = np.array(frames[-1]["grid"], dtype=np.int32)

        feats = compute_feature_maps(ig_raw)
        feat_tensor = features_to_tensor(feats)
        fg_classes = code_to_class_grid(fg_raw)

        feats_list.append(feat_tensor)
        finals_list.append(fg_classes)
        statics_list.append(feats['static'])

    n = len(feats_list)
    print(f"  {n} replays loaded")

    feat_t = torch.from_numpy(np.stack(feats_list)).to(DEVICE)       # (N, 12, H, W)
    final_t = torch.from_numpy(np.stack(finals_list)).long().to(DEVICE)  # (N, H, W)
    static_t = torch.from_numpy(np.stack(statics_list)).bool().to(DEVICE)  # (N, H, W)

    # Augment with rotations and flips (8x)
    all_f, all_fg, all_s = [feat_t], [final_t], [static_t]
    for k in range(1, 4):
        all_f.append(torch.rot90(feat_t, k, [2, 3]))
        all_fg.append(torch.rot90(final_t, k, [1, 2]))
        all_s.append(torch.rot90(static_t, k, [1, 2]))

    feat_aug = torch.cat(all_f)
    final_aug = torch.cat(all_fg)
    static_aug = torch.cat(all_s)

    # Flips
    feat_aug = torch.cat([feat_aug, torch.flip(feat_aug, [3])])
    final_aug = torch.cat([final_aug, torch.flip(final_aug, [2])])
    static_aug = torch.cat([static_aug, torch.flip(static_aug, [2])])

    print(f"  {feat_aug.shape[0]} samples after augmentation")
    return feat_aug, final_aug, static_aug


# ---------------------------------------------------------------------------
# Model: ConvNet with local features + spatial context
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class FeatureConditionedD3PM(nn.Module):
    """Predicts per-cell class distribution conditioned on local features.

    Input: features (12ch) + noised_grid one-hot (7ch) + timestep (1ch) = 20ch
    Output: (6, H, W) logits

    Uses dilated convs for multi-scale spatial context (critical for CA dynamics).
    """

    def __init__(self, base_ch=64):
        super().__init__()
        in_ch = N_FEAT_CH + N_TOKENS + 1  # 12 + 7 + 1 = 20

        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.blocks = nn.ModuleList([
            ResBlock(base_ch),
            ResBlock(base_ch),
            ResBlock(base_ch),
            ResBlock(base_ch),
            ResBlock(base_ch),
            ResBlock(base_ch),
            ResBlock(base_ch),
            ResBlock(base_ch),
        ])

        # Multi-scale context via dilated convolutions
        self.dilated = nn.ModuleList([
            nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=2, dilation=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=4, dilation=4), nn.SiLU()),
            nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=8, dilation=8), nn.SiLU()),
            nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=16, dilation=16), nn.SiLU()),
        ])

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, N_CLASSES, 1),
        )

    def forward(self, x):
        h = self.stem(x)

        h = self.blocks[0](h)
        h = self.blocks[1](h)
        h = h + self.dilated[0](h)

        h = self.blocks[2](h)
        h = self.blocks[3](h)
        h = h + self.dilated[1](h)

        h = self.blocks[4](h)
        h = self.blocks[5](h)
        h = h + self.dilated[2](h)

        h = self.blocks[6](h)
        h = self.blocks[7](h)
        h = h + self.dilated[3](h)

        return self.out(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def apply_noise_batch(fgs, t_values, T):
    B = fgs.shape[0]
    mask_probs = t_values.float() / T
    rand = torch.rand(B, H, W, device=fgs.device)
    mask = rand < mask_probs[:, None, None]
    noised = fgs.clone()
    noised[mask] = MASK_TOKEN
    return F.one_hot(noised, N_TOKENS).permute(0, 3, 1, 2).float()


def train(epochs=500, batch_size=128, lr=3e-4, T=20, base_ch=64):
    print("Loading and precomputing dataset...")
    feats, finals, statics = precompute_dataset()
    N = feats.shape[0]

    model = FeatureConditionedD3PM(base_ch=base_ch).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params on {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            B = len(idx)

            feat_batch = feats[idx]       # (B, 12, H, W)
            fg_batch = finals[idx]        # (B, H, W)
            static_batch = statics[idx]   # (B, H, W)

            # Random timesteps
            t_batch = torch.randint(1, T + 1, (B,), device=DEVICE)

            # Apply noise to final grid
            noised_batch = apply_noise_batch(fg_batch, t_batch, T)  # (B, 7, H, W)

            # Timestep channel
            t_channel = (t_batch.float() / T)[:, None, None, None].expand(B, 1, H, W)

            # Concat: features + noised grid + timestep
            x = torch.cat([feat_batch, noised_batch, t_channel], dim=1)  # (B, 20, H, W)

            logits = model(x)  # (B, 6, H, W)

            # Loss: weighted cross-entropy (ignore static cells)
            loss = F.cross_entropy(logits, fg_batch, reduction='none')
            weight = torch.ones_like(loss)
            weight[static_batch] = 0.0  # zero out static cells entirely
            loss = (loss * weight).sum() / weight.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}, time={elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state': model.state_dict(),
                'base_ch': base_ch,
                'T': T,
                'epoch': epoch,
                'loss': avg_loss,
            }, BASE_DIR / "diffusion_ca_best.pt")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Best loss: {best_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Inference: batched denoising for speed
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise_batch(model, feat_tensor, ig_raw, T=20, n_samples=50, temperature=1.0):
    """Run n_samples denoising trajectories IN PARALLEL.

    feat_tensor: (12, H, W) precomputed features
    Returns: (n_samples, H, W) sampled class grids
    """
    static = (ig_raw == OCEAN) | (ig_raw == MOUNTAIN)
    ig_classes = code_to_class_grid(ig_raw)
    static_t = torch.from_numpy(static).to(DEVICE)
    ig_t = torch.from_numpy(ig_classes).long().to(DEVICE)

    B = n_samples
    feat_batch = feat_tensor.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 12, H, W)

    # Start fully masked
    current = torch.full((B, H, W), MASK_TOKEN, dtype=torch.long, device=DEVICE)
    current[:, static_t] = ig_t[static_t].unsqueeze(0).expand(B, -1)

    for t_idx in range(T, 0, -1):
        noised_onehot = F.one_hot(current, N_TOKENS).permute(0, 3, 1, 2).float()
        t_channel = torch.full((B, 1, H, W), t_idx / T, device=DEVICE)
        x = torch.cat([feat_batch, noised_onehot, t_channel], dim=1)

        logits = model(x)  # (B, 6, H, W)
        probs = F.softmax(logits / temperature, dim=1)  # (B, 6, H, W)

        # How many cells to unmask
        currently_masked = (current == MASK_TOKEN)  # (B, H, W)
        target_frac = (t_idx - 1) / T

        # Per-sample: determine cells to unmask
        max_probs = probs.max(dim=1).values  # (B, H, W)

        for b in range(B):
            b_masked = currently_masked[b]
            n_masked = b_masked.sum().item()
            # Count non-static cells
            n_dynamic = (~static_t).sum().item()
            target_masked = int(target_frac * n_dynamic)
            n_to_unmask = max(0, n_masked - target_masked)

            if n_to_unmask > 0 and n_masked > 0:
                conf = torch.where(b_masked, max_probs[b], torch.tensor(-1.0, device=DEVICE))
                gumbel = -torch.log(-torch.log(torch.rand_like(conf) + 1e-8) + 1e-8) * 0.1
                conf = conf + gumbel

                _, top_idx = conf.view(-1).topk(min(n_to_unmask, n_masked))

                for idx in top_idx:
                    y, xc = idx // W, idx % W
                    sampled = torch.multinomial(probs[b, :, y, xc], 1).item()
                    current[b, y, xc] = sampled

    # Final cleanup
    remaining = (current == MASK_TOKEN)
    if remaining.any():
        noised_onehot = F.one_hot(current, N_TOKENS).permute(0, 3, 1, 2).float()
        t_channel = torch.full((B, 1, H, W), 1 / T, device=DEVICE)
        x = torch.cat([feat_batch, noised_onehot, t_channel], dim=1)
        logits = model(x)

        for b in range(B):
            rem = remaining[b]
            if rem.any():
                for y, xc in torch.argwhere(rem):
                    current[b, y, xc] = torch.multinomial(
                        F.softmax(logits[b, :, y, xc] / temperature, dim=0), 1).item()

    # Enforce static
    current[:, static_t] = ig_t[static_t].unsqueeze(0).expand(B, -1)

    return current.cpu().numpy()


@torch.no_grad()
def predict(initial_grid, n_samples=200, T=20, temperature=1.0, model=None):
    """Generate probability tensor from multiple denoising samples."""
    if model is None:
        ckpt = torch.load(BASE_DIR / "diffusion_ca_best.pt", map_location=DEVICE, weights_only=True)
        model = FeatureConditionedD3PM(base_ch=ckpt.get('base_ch', 64)).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        T = ckpt.get('T', T)
    model.eval()

    ig = np.array(initial_grid, dtype=np.int32)
    feats = compute_feature_maps(ig)
    feat_tensor = torch.from_numpy(features_to_tensor(feats)).to(DEVICE)

    # Run batched denoising (process in chunks to fit in GPU memory)
    chunk_size = min(50, n_samples)
    counts = np.zeros((H, W, N_CLASSES), dtype=np.float64)

    remaining = n_samples
    while remaining > 0:
        batch = min(chunk_size, remaining)
        samples = denoise_batch(model, feat_tensor, ig, T=T,
                                n_samples=batch, temperature=temperature)
        for c in range(N_CLASSES):
            counts[:, :, c] += (samples == c).sum(axis=0)
        remaining -= batch

    probs = counts / n_samples

    FLOOR = 1e-6
    probs = np.maximum(probs, FLOOR)
    probs /= probs.sum(axis=2, keepdims=True)

    probs[ig == OCEAN] = [1, 0, 0, 0, 0, 0]
    probs[ig == MOUNTAIN] = [0, 0, 0, 0, 0, 1]

    return probs


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def quick_eval(model, T=20, n_samples=50, n_seeds=10):
    gt_files = sorted(GT_DIR.glob("round*_seed*.json"))[:n_seeds]
    wkls = []
    for f in gt_files:
        data = json.loads(f.read_text())
        ig = np.array(data["initial_grid"], dtype=np.int32)
        gt = np.array(data["ground_truth"], dtype=np.float64)

        pred = predict(ig, n_samples=n_samples, T=T, model=model)

        gt_safe = np.maximum(gt, 1e-10)
        pred_safe = np.maximum(pred, 1e-6)
        pred_safe /= pred_safe.sum(axis=2, keepdims=True)

        kl = np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=2)
        entropy = -np.sum(gt_safe * np.log2(gt_safe + 1e-10), axis=2)
        wkl = entropy * kl
        dynamic = entropy > 0.01

        mean_wkl = float(wkl[dynamic].mean()) if dynamic.any() else 0.0
        score = max(0, min(100, 100 * math.exp(-3 * mean_wkl)))
        wkls.append(mean_wkl)
        print(f"  {f.stem}: wKL={mean_wkl:.4f} score={score:.1f}")

    overall = np.mean(wkls)
    print(f"\n  Mean wKL={overall:.4f} score={max(0, min(100, 100*math.exp(-3*overall))):.1f}")
    return overall


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-eval-seeds", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    if args.train:
        print("=" * 60)
        print("  D3PM v2: Feature-conditioned Discrete Diffusion")
        print("=" * 60)
        model = train(
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, T=args.T, base_ch=args.base_ch,
        )
        if args.eval:
            print("\n" + "=" * 60)
            print("  Quick Evaluation")
            print("=" * 60)
            quick_eval(model, T=args.T, n_samples=args.n_samples,
                       n_seeds=args.n_eval_seeds)

    elif args.eval:
        ckpt = torch.load(BASE_DIR / "diffusion_ca_best.pt", map_location=DEVICE, weights_only=True)
        model = FeatureConditionedD3PM(base_ch=ckpt.get('base_ch', 64)).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"Loaded epoch {ckpt.get('epoch','?')}, loss={ckpt.get('loss','?'):.4f}")
        quick_eval(model, T=ckpt.get('T', args.T), n_samples=args.n_samples,
                   n_seeds=args.n_eval_seeds)
