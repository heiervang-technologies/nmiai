import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))
import regime_predictor
try:
    import benchmark_experiments
except ImportError:
    benchmark_experiments = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_wkl(pred, target, ig):
    dynamic_mask = (ig != 10) & (ig != 5)
    p = np.clip(target, 1e-8, 1.0)
    q = np.clip(pred, 1e-8, 1.0)
    kl = (p * (np.log(p) - np.log(q))).sum(axis=-1)
    entropy_bits = -(p * np.log2(p)).sum(axis=-1)
    weights = (0.15 + entropy_bits) * dynamic_mask
    return np.sum(weights * kl) / np.sum(weights)

def run_gpu_mc(ig, interpolated_lookup, num_sims=2000):
    H, W = ig.shape
    has_ruin_dim = interpolated_lookup.dim() >= 8  # v2 tensor has r3 dimension

    # Kernels
    k3 = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
    k3[0, 0, 1, 1] = 0
    k7 = torch.ones((1, 1, 7, 7), device=device, dtype=torch.float32)
    k7[0, 0, 3, 3] = 0

    state = torch.tensor(ig, device=device, dtype=torch.long).unsqueeze(0).repeat(num_sims, 1, 1)

    # Handle raw tensor with regime dimension
    if interpolated_lookup.dim() == 8 and interpolated_lookup.shape[0] == 3 and interpolated_lookup.shape[-1] == 12 and interpolated_lookup.shape[1] == 4:
        # v1 raw [3, 4, 12, 9, 26, 9, 9, 12] - blend regimes
        interpolated_lookup = (interpolated_lookup[0] + interpolated_lookup[1] + interpolated_lookup[2]) / 3.0
        has_ruin_dim = False
    elif interpolated_lookup.dim() == 9 and interpolated_lookup.shape[0] == 3:
        # v2 raw [3, 4, 12, 9, 26, 9, 9, 4, 12] - blend regimes
        interpolated_lookup = (interpolated_lookup[0] + interpolated_lookup[1] + interpolated_lookup[2]) / 3.0
        has_ruin_dim = True
    elif interpolated_lookup.dim() == 8 and interpolated_lookup.shape[-1] == 12 and interpolated_lookup.shape[0] == 4:
        # v2 interpolated [4, 12, 9, 26, 9, 9, 4, 12]
        has_ruin_dim = True
    else:
        # v1 interpolated [4, 12, 9, 26, 9, 9, 12]
        has_ruin_dim = False

    for step in range(1, 51):
        phase = step % 4

        is_civ = ((state == 1) | (state == 2)).float().unsqueeze(1)
        n_civ3 = F.conv2d(is_civ, k3, padding=1).squeeze(1).long()
        n_civ3 = torch.clamp(n_civ3, 0, 8)

        n_civ7 = F.conv2d(is_civ, k7, padding=3).squeeze(1).long()
        n_civ7 = torch.clamp(n_civ7, 0, 25)

        is_ocean = (state == 10).float().unsqueeze(1)
        n_ocean = F.conv2d(is_ocean, k3, padding=1).squeeze(1).long()
        n_ocean = torch.clamp(n_ocean, 0, 8)

        is_forest = (state == 4).float().unsqueeze(1)
        n_forest = F.conv2d(is_forest, k3, padding=1).squeeze(1).long()
        n_forest = torch.clamp(n_forest, 0, 8)

        phase_lookup = interpolated_lookup[phase]
        flat_state = torch.clamp(state.view(-1), 0, 11)
        flat_c3 = torch.clamp(n_civ3.view(-1), 0, 8)
        flat_c7 = torch.clamp(n_civ7.view(-1), 0, 25)
        flat_o3 = torch.clamp(n_ocean.view(-1), 0, 8)
        flat_f3 = torch.clamp(n_forest.view(-1), 0, 8)

        if has_ruin_dim:
            is_ruin = (state == 3).float().unsqueeze(1)
            n_ruin3 = F.conv2d(is_ruin, k3, padding=1).squeeze(1).long()
            flat_r3 = torch.clamp(n_ruin3.view(-1), 0, 3)
            probs = phase_lookup[flat_state, flat_c3, flat_c7, flat_o3, flat_f3, flat_r3]
        else:
            probs = phase_lookup[flat_state, flat_c3, flat_c7, flat_o3, flat_f3]

        # Safety: ensure valid probability distribution for multinomial
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        sampled = torch.multinomial(probs, 1).squeeze(-1)
        state = sampled.view(num_sims, H, W)
        
    final_states = state.cpu().numpy()
    pred = np.zeros((H, W, 6), dtype=np.float32)
    for c in range(12):
        if c in (0, 10, 11): target_c = 0
        elif c in (1, 2, 3, 4, 5): target_c = c
        else: target_c = 0
        
        mask = (final_states == c).mean(axis=0)
        pred[..., target_c] += mask
        
    return pred

def evaluate_blend(blend_w, lookup_tensor, ig, obs_list, num_sims=100):
    # interpolate
    interp_tensor = blend_w[0] * lookup_tensor[0] + blend_w[1] * lookup_tensor[1] + blend_w[2] * lookup_tensor[2]
    pred = run_gpu_mc(ig, interp_tensor, num_sims=num_sims)
    
    # Calculate Log-Likelihood of the observations given this prediction
    ll = 0.0
    for obs in obs_list:
        ox, oy = obs['viewport_x'], obs['viewport_y']
        obs_grid = np.array(obs['grid'])
        oh, ow = obs_grid.shape
        
        # pred slice
        pred_slice = pred[oy:oy+oh, ox:ox+ow] # shape (h, w, 6)
        
        # We need to map the exact observed classes (0,1,2,3,4,5,10,11) to the 6 output classes
        for y in range(oh):
            for x in range(ow):
                c = obs_grid[y, x]
                if c in (0, 10, 11): target_c = 0
                elif c in (1, 2, 3, 4, 5): target_c = c
                else: target_c = 0
                
                prob = pred_slice[y, x, target_c]
                prob = max(prob, 1e-6)
                ll += np.log(prob)
                
    return ll

def eval_gpu_mc():
    print("Loading dense tensor to GPU...")
    lookup_tensor = torch.load('tasks/astar-island/dense_lookup.pt').to(device)
    files = glob.glob('tasks/astar-island/ground_truth/*.json')
    
    total_wkl = 0
    count = 0
    
    print("Running Test-Time-Optimized GPU Monte-Carlo Benchmark...")
    t0 = time.time()
    
    for f in files:
        basename = os.path.basename(f)
        rn = int(basename.split('_')[0].replace('round', ''))
        sn = int(basename.split('_')[1].replace('seed', '').replace('.json', ''))
        
        if rn < 1:
            continue
            
        with open(f) as fh:
            d = json.load(fh)
        
        ig = np.array(d['initial_grid'], dtype=np.int32)
        gt = np.array(d['ground_truth'], dtype=np.float32)
        
        # 1. Gather observation queries
        obs = benchmark_experiments.simulate_observations_from_gt(gt, ig, n_viewports=3, strategy="hotspot")
        
        # 2. Test-Time Optimization (TTO)
        # We will grid search across a simplex of (Harsh, Moderate, Prosperous)
        best_ll = -99999999
        best_w = (0, 1, 0)
        
        # Generate simplex blends
        blends = []
        for p in np.linspace(0, 1, 11):
            for m in np.linspace(0, 1 - p, 11):
                h = 1.0 - p - m
                blends.append((h, m, p))
                
        # Downsample to 10 logical points for speed
        blends = [(1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0,0.2,0.8), (0,0.8,0.2), (0.2,0.8,0), (0.8,0.2,0), (0.33,0.33,0.34)]
        
        for w in blends:
            ll = evaluate_blend(w, lookup_tensor, ig, obs, num_sims=200)
            if ll > best_ll:
                best_ll = ll
                best_w = w
        
        # 3. Final Massive Compute Run
        interpolated_tensor = best_w[0] * lookup_tensor[0] + best_w[1] * lookup_tensor[1] + best_w[2] * lookup_tensor[2]
        pred = run_gpu_mc(ig, interpolated_tensor, num_sims=3000)
        
        # Structural Zeros
        pred = np.maximum(pred, 1e-8)
        pred[ig == 5] = [0, 0, 0, 0, 0, 1]
        pred[ig == 10] = [1, 0, 0, 0, 0, 0]
        
        from scipy.ndimage import distance_transform_edt
        mask = (ig == 1) | (ig == 2)
        dist = distance_transform_edt(~mask) if mask.any() else np.full_like(ig, 100)
        pred[(dist > 12) & ((ig == 1) | (ig == 2) | (ig == 0) | (ig == 11))] = [1, 0, 0, 0, 0, 0]
        pred[(dist > 12) & (ig == 4)] = [0, 0, 0, 0, 1, 0]
        
        # Quantization-Aware Snapping Trick
        target_sum = 200
        counts = np.floor(pred * target_sum).astype(int)
        remainders = (pred * target_sum) - counts
        deficiency = target_sum - counts.sum(axis=-1)
        
        for i in range(40):
            for j in range(40):
                if deficiency[i, j] > 0:
                    idx = np.argsort(remainders[i, j])[-deficiency[i, j]:]
                    counts[i, j, idx] += 1
        
        snapped_prob = counts / 200.0
        snapped_prob = np.maximum(snapped_prob, 1e-6)
        snapped_prob /= snapped_prob.sum(axis=-1, keepdims=True)
        
        wkl = calc_wkl(snapped_prob, gt, ig)
        print(f"R{rn} S{sn} [TTO Blend H:{best_w[0]:.2f} M:{best_w[1]:.2f} P:{best_w[2]:.2f}]: wKL = {wkl:.4f}")
        total_wkl += wkl
        count += 1
        
    t1 = time.time()
    print(f"\nFinal TTO GPU MC wKL: {total_wkl/count:.4f} (Time: {t1-t0:.2f}s)")

if __name__ == '__main__':
    eval_gpu_mc()
