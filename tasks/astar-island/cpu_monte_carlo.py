import json
import glob
import numpy as np
import sys
import os
from scipy.ndimage import convolve

sys.path.append(os.path.dirname(__file__))
import regime_predictor
import benchmark_experiments

def load_tables():
    with open('tasks/astar-island/mc_transition_tables.json') as f:
        data = json.load(f)
    
    # We want a fast lookup array: table[regime_idx, phase, state, n_civ, next_state] = prob
    # regime: 0=Harsh, 1=Moderate, 2=Prosperous
    # phase: 0..3
    # state: 0..11
    # n_civ: 0..8
    # next_state: 0..11
    lookup = np.zeros((3, 4, 12, 9, 12), dtype=np.float32)
    
    r_map = {'Harsh': 0, 'Moderate': 1, 'Prosperous': 2}
    
    # For states we didn't track, just stay the same (prob 1.0 for next_state == state)
    for r in range(3):
        for p in range(4):
            for st in range(12):
                for cv in range(9):
                    lookup[r, p, st, cv, st] = 1.0
                    
    for r_str, r_data in data.items():
        r_idx = r_map.get(r_str)
        if r_idx is None: continue
        for p_str, p_data in r_data.items():
            p_idx = int(p_str)
            for st_str, st_data in p_data.items():
                st_idx = int(st_str)
                for cv_str, cv_data in st_data.items():
                    cv_idx = int(cv_str)
                    
                    # Clear the default
                    lookup[r_idx, p_idx, st_idx, cv_idx, st_idx] = 0.0
                    
                    total_p = 0.0
                    for nxt_str, prob in cv_data.items():
                        nxt_idx = int(nxt_str)
                        lookup[r_idx, p_idx, st_idx, cv_idx, nxt_idx] = prob
                        total_p += prob
                        
                    # Handle rounding errors or missing mass
                    if total_p > 0:
                        lookup[r_idx, p_idx, st_idx, cv_idx] /= np.sum(lookup[r_idx, p_idx, st_idx, cv_idx])
                    else:
                        lookup[r_idx, p_idx, st_idx, cv_idx, st_idx] = 1.0
                        
    return lookup

def calc_wkl(pred, target, ig):
    dynamic_mask = (ig != 10) & (ig != 5)
    p = np.clip(target, 1e-8, 1.0)
    q = np.clip(pred, 1e-8, 1.0)
    kl = (p * (np.log(p) - np.log(q))).sum(axis=-1)
    entropy_bits = -(p * np.log2(p)).sum(axis=-1)
    weights = (0.15 + entropy_bits) * dynamic_mask
    return np.sum(weights * kl) / np.sum(weights)

def simulate_cpu_mc(ig, regime_idx, lookup, num_sims=200):
    """Run 200 MC simulations on CPU with numpy."""
    h, w = ig.shape
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)
    
    final_states = np.zeros((num_sims, h, w), dtype=np.int32)
    
    for sim in range(num_sims):
        grid = ig.copy()
        
        for step in range(1, 51):
            phase = step % 4
            is_civ = ((grid == 1) | (grid == 2)).astype(np.int32)
            n_civ = convolve(is_civ, kernel, mode='constant')
            
            # Map probabilities for the entire grid
            # lookup shape: (3, 4, 12, 9, 12) -> we want to index with [regime_idx, phase, grid, n_civ]
            probs = lookup[regime_idx, phase, grid, n_civ] # shape (40, 40, 12)
            
            # Fast vectorized categorical sampling
            # probs shape is (40, 40, 12)
            # cumsum to get CDF, then compare with random float
            cdf = np.cumsum(probs, axis=-1)
            r = np.random.rand(h, w, 1)
            next_grid = (r > cdf).sum(axis=-1).astype(np.int32)
            next_grid = np.clip(next_grid, 0, 11) # safety bounds
            
            grid = next_grid
            
        final_states[sim] = grid
        
    # Convert to 6-class probability distribution
    # Class map: 0->0, 1->1, 2->2, 3->3, 4->4, 5->5, 10->0, 11->0
    pred = np.zeros((h, w, 6), dtype=np.float32)
    for c in range(12):
        if c in (0, 10, 11): target_c = 0
        elif c in (1, 2, 3, 4, 5): target_c = c
        else: target_c = 0
        
        mask = (final_states == c).mean(axis=0)
        pred[..., target_c] += mask
        
    return pred

def eval_baseline():
    lookup = load_tables()
    files = glob.glob('tasks/astar-island/ground_truth/*.json')
    
    print("Evaluating CPU Monte-Carlo Baseline on R15-R17...")
    total_wkl = 0
    count = 0
    
    r_map = {'harsh': 0, 'moderate': 1, 'prosperous': 2}
    
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
        
        # Simulate observations to detect regime (like a real run)
        obs = benchmark_experiments.simulate_observations_from_gt(gt, ig, n_viewports=3, strategy="hotspot")
        r_weights_dict = regime_predictor.detect_regime_from_observations(ig, obs)
        
        # Pick the most likely regime
        best_regime = max(r_weights_dict, key=r_weights_dict.get)
        regime_idx = r_map[best_regime]
        
        pred = simulate_cpu_mc(ig, regime_idx, lookup, num_sims=200)
        
        # Safe bets: floor and structural zeros
        pred = np.maximum(pred, 1e-6)
        pred[ig == 5] = [0, 0, 0, 0, 0, 1]
        pred[ig == 10] = [1, 0, 0, 0, 0, 0]
        
        pred /= pred.sum(axis=-1, keepdims=True)
        
        wkl = calc_wkl(pred, gt, ig)
        print(f"R{rn} S{sn} ({best_regime}): wKL = {wkl:.4f}")
        total_wkl += wkl
        count += 1
        
    print(f"\nFinal CPU MC Baseline wKL: {total_wkl/count:.4f}")

if __name__ == '__main__':
    eval_baseline()