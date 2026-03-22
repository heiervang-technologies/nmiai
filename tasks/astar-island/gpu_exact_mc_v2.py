import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import time

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

def run_gpu_mc(ig, db, regime, num_sims=2000):
    H, W = ig.shape
    keys_db = db['keys'].to(device)
    probs_db = db['probs'].to(device)
    m_ph, m_rg, m_ce, m_c3, m_c7, m_o3, m_f3, m_r3, m_p3 = db['multipliers']
    
    k3 = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
    k3[0, 0, 1, 1] = 0
    k7 = torch.ones((1, 1, 7, 7), device=device, dtype=torch.float32)
    k7[0, 0, 3, 3] = 0
    
    state = torch.tensor(ig, device=device, dtype=torch.long).unsqueeze(0).repeat(num_sims, 1, 1)
    
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
        
        is_ruin = (state == 3).float().unsqueeze(1)
        n_ruin = F.conv2d(is_ruin, k3, padding=1).squeeze(1).long()
        n_ruin = torch.clamp(n_ruin, 0, 8)
        
        is_port = (state == 2).float().unsqueeze(1)
        n_port = F.conv2d(is_port, k3, padding=1).squeeze(1).long()
        n_port = torch.clamp(n_port, 0, 8)
        
        flat_state = torch.clamp(state.view(-1), 0, 11)
        
        # Compute flat integer keys
        query_keys = (phase * m_ph + 
                      regime * m_rg + 
                      flat_state * m_ce + 
                      n_civ3.view(-1) * m_c3 + 
                      n_civ7.view(-1) * m_c7 + 
                      n_ocean.view(-1) * m_o3 + 
                      n_forest.view(-1) * m_f3 + 
                      n_ruin.view(-1) * m_r3 + 
                      n_port.view(-1) * m_p3)
                      
        # Binary search for exact match
        idx = torch.searchsorted(keys_db, query_keys)
        # Cap index to size
        idx = torch.clamp(idx, 0, keys_db.size(0) - 1)
        
        # Check if matched
        matched = (keys_db[idx] == query_keys)
        
        # If not matched, point to fallback index 0 (which we handle below)
        idx = torch.where(matched, idx, torch.zeros_like(idx))
        
        probs = probs_db[idx] # [B*H*W, 12]
        
        # Fallback: if not matched, probability is 1.0 to stay in current state
        # We scatter 1.0 into the one-hot center state
        fallback_probs = F.one_hot(flat_state, num_classes=12).float()
        probs = torch.where(matched.unsqueeze(-1), probs, fallback_probs)
        
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

def eval_gpu_mc():
    print("Loading search tensor to GPU...")
    db = torch.load('tasks/astar-island/gpu_search_tensor.pt', weights_only=False)
    
    files = glob.glob('tasks/astar-island/ground_truth/*.json')
    
    total_wkl = 0
    count = 0
    
    print("Running GPU Monte-Carlo Benchmark V2...")
    t0 = time.time()
    
    for f in files:
        basename = os.path.basename(f)
        rn = int(basename.split('_')[0].replace('round', ''))
        sn = int(basename.split('_')[1].replace('seed', '').replace('.json', ''))
        
        if rn < 15 or rn > 17:
            continue
            
        with open(f) as fh:
            d = json.load(fh)
        
        ig = np.array(d['initial_grid'], dtype=np.int32)
        gt = np.array(d['ground_truth'], dtype=np.float32)
        
        # 1. Gather observation queries
        obs = benchmark_experiments.simulate_observations_from_gt(gt, ig, n_viewports=3, strategy="hotspot")
        
        # 2. TTO for Regime (Discrete search instead of blending)
        best_ll = -99999999
        best_regime = 1
        
        for regime in [0, 1, 2]:
            pred = run_gpu_mc(ig, db, regime, num_sims=200)
            
            ll = 0.0
            for ob in obs:
                ox, oy = ob['viewport_x'], ob['viewport_y']
                obs_grid = np.array(ob['grid'])
                oh, ow = obs_grid.shape
                pred_slice = pred[oy:oy+oh, ox:ox+ow]
                
                for y in range(oh):
                    for x in range(ow):
                        c = obs_grid[y, x]
                        if c in (0, 10, 11): target_c = 0
                        elif c in (1, 2, 3, 4, 5): target_c = c
                        else: target_c = 0
                        
                        prob = pred_slice[y, x, target_c]
                        prob = max(prob, 1e-6)
                        ll += np.log(prob)
                        
            if ll > best_ll:
                best_ll = ll
                best_regime = regime
        
        # 3. Final Massive Compute Run
        pred = run_gpu_mc(ig, db, best_regime, num_sims=3000)
        
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
        reg_str = ["Harsh", "Moderate", "Prosperous"][best_regime]
        print(f"R{rn} S{sn} [TTO Regime: {reg_str}]: wKL = {wkl:.4f}")
        total_wkl += wkl
        count += 1
        
    t1 = time.time()
    print(f"\nFinal TTO GPU MC V2 wKL: {total_wkl/count:.4f} (Time: {t1-t0:.2f}s)")

if __name__ == '__main__':
    eval_gpu_mc()
