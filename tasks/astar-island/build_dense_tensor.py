import json
import glob
import numpy as np
import torch
from scipy.ndimage import convolve
import sys

def get_regime(final_grid):
    civ_count = np.sum((final_grid == 1) | (final_grid == 2))
    if civ_count < 30: return 0 # Harsh
    if civ_count > 150: return 2 # Prosperous
    return 1 # Moderate

def build_dense_tensor():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files += glob.glob('tasks/astar-island/replays/dense_training/*.json')
    # Filter unique replays just in case
    files = list(set(files))
    
    print(f"Building dense transition tensor from {len(files)} replays...")

    k3 = np.ones((3,3), dtype=np.int32)
    k3[1,1] = 0
    k7 = np.ones((7,7), dtype=np.int32)
    k7[3,3] = 0

    # Dimensions
    # R=3, P=4, ST=12, C3=9, C7=26 (capped at 25), O3=9, F3=9, NXT=12
    counts_full = np.zeros((3, 4, 12, 9, 26, 9, 9, 12), dtype=np.float32)
    
    for idx, f in enumerate(files):
        if idx % 50 == 0:
            print(f"Processing file {idx}/{len(files)}...")
            
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 2: continue
        
        regime = get_regime(np.array(frames[-1]['grid']))
        
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            phase = i % 4
            
            is_civ = ((prev == 1) | (prev == 2)).astype(np.int32)
            n_civ3 = convolve(is_civ, k3, mode='constant')
            n_civ7 = convolve(is_civ, k7, mode='constant')
            n_civ7 = np.clip(n_civ7, 0, 25) # Cap at 25
            
            is_ocean = (prev == 10).astype(np.int32)
            n_ocean = convolve(is_ocean, k3, mode='constant')
            
            is_forest = (prev == 4).astype(np.int32)
            n_forest = convolve(is_forest, k3, mode='constant')
            
            # Vectorized counting
            # We only care about dynamic cells to save processing time
            # Actually, to be perfect, let's just bin count using np.add.at
            diff = prev != curr
            
            # To avoid adding 1600 cells per frame which takes forever in pure python loops,
            # we will flatten and use np.add.at
            valid_mask = ~((prev == 0) | (prev == 5) | (prev == 10) | (prev == 11))
            # Wait, Plains (11) and Empty (0) DO change.
            # Mountains(5) and Oceans(10) don't change.
            valid_mask = ~((prev == 5) | (prev == 10))
            
            # Actually we can just do it for all cells that changed, AND a sample of cells that didn't change.
            changed_y, changed_x = np.where(diff & valid_mask)
            
            for y, x in zip(changed_y, changed_x):
                st = prev[y,x]
                nxt = curr[y,x]
                c3 = n_civ3[y,x]
                c7 = n_civ7[y,x]
                o3 = n_ocean[y,x]
                f3 = n_forest[y,x]
                counts_full[regime, phase, st, c3, c7, o3, f3, nxt] += 1
                
            # Sample of unchanged cells
            stayed = (~diff) & valid_mask
            if np.any(stayed):
                sy, sx = np.where(stayed)
                if len(sy) > 200:
                    indices = np.random.choice(len(sy), 200, replace=False)
                    sy = sy[indices]
                    sx = sx[indices]
                    
                multiplier = np.sum(stayed) / len(sy) # Scale up to true count
                for y, x in zip(sy, sx):
                    st = prev[y,x]
                    nxt = curr[y,x]
                    c3 = n_civ3[y,x]
                    c7 = n_civ7[y,x]
                    o3 = n_ocean[y,x]
                    f3 = n_forest[y,x]
                    counts_full[regime, phase, st, c3, c7, o3, f3, nxt] += multiplier

    print("Aggregating probabilities and handling sparsity fallback...")
    
    # Base fallback table (Regime, Phase, State, C3)
    # Sum across C7, O3, F3
    counts_base = np.sum(counts_full, axis=(4, 5, 6))
    sums_base = np.sum(counts_base, axis=4, keepdims=True)
    probs_base = np.divide(counts_base, sums_base, out=np.zeros_like(counts_base), where=sums_base!=0)
    
    # If a state has 0 occurrences in base, it stays itself
    for r in range(3):
        for p in range(4):
            for st in range(12):
                for c3 in range(9):
                    if sums_base[r, p, st, c3, 0] == 0:
                        probs_base[r, p, st, c3, st] = 1.0
                        
    # Now build the full probability tensor
    sums_full = np.sum(counts_full, axis=7, keepdims=True)
    probs_full = np.divide(counts_full, sums_full, out=np.zeros_like(counts_full), where=sums_full!=0)
    
    # Fill in sparse entries with the base fallback
    MIN_SAMPLES = 10
    sparse_mask = (sums_full < MIN_SAMPLES).squeeze(-1) # [3, 4, 12, 9, 26, 9, 9]
    
    probs_base_bcast = np.broadcast_to(np.expand_dims(probs_base, axis=(4,5,6)), counts_full.shape)
    probs_full[sparse_mask] = probs_base_bcast[sparse_mask]
    
    # Clean up static states
    for st in [5, 10]:
        probs_full[:, :, st, ...] = 0.0
        probs_full[:, :, st, ..., st] = 1.0

    # STRICT STRUCTURAL CONSTRAINTS (from exact_transition_rules.md)
    # 1. Ruins (3) last exactly 1 step (can NEVER transition to Ruin)
    probs_full[:, :, 3, ..., 3] = 0.0
    
    # 2. Empty (0) and Plains (11) can NEVER directly become Forest (4)
    probs_full[:, :, 0, ..., 4] = 0.0
    probs_full[:, :, 11, ..., 4] = 0.0
    
    # 3. Forest (4) can NEVER directly become Empty (0) or Plains (11)
    probs_full[:, :, 4, ..., 0] = 0.0
    probs_full[:, :, 4, ..., 11] = 0.0

    # 4. Port (2) can NEVER form or exist if there are 0 Ocean neighbors (o3 == 0)
    # The o3 dimension is index 5 in the array (R=3, P=4, ST=12, C3=9, C7=26, O3=9, F3=9, NXT=12)
    probs_full[:, :, :, :, :, 0, :, 2] = 0.0
    
    # Renormalize to ensure sum to 1.0 after applying zero-constraints
    row_sums = np.sum(probs_full, axis=7, keepdims=True)
    probs_full = np.divide(probs_full, row_sums, out=np.zeros_like(probs_full), where=row_sums!=0)
    
    # Fallback for any rows that were completely zeroed out (should be rare)
    zero_rows = (row_sums.squeeze(-1) == 0)
    # If a state was zeroed out completely, just default it to staying the same state
    for st in range(12):
        if np.any(zero_rows[:, :, st, ...]):
            mask = zero_rows[:, :, st, ...]
            probs_full[:, :, st, ..., st][mask] = 1.0

    print("Saving dense_lookup.pt...")
    torch.save(torch.tensor(probs_full, dtype=torch.float32), 'tasks/astar-island/dense_lookup.pt')
    print("Done! Tensor size:", probs_full.nbytes / 1024 / 1024, "MB")

if __name__ == '__main__':
    build_dense_tensor()
