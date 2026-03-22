import json
import glob
import numpy as np
import torch
import time
import os
import gpu_exact_mc_v2
import benchmark_experiments

def eval_all():
    print("Loading search tensor to GPU...")
    db = torch.load('tasks/astar-island/gpu_search_tensor.pt', weights_only=False)
    
    files = glob.glob('tasks/astar-island/ground_truth/*.json')
    
    total_wkl = 0
    count = 0
    
    print(f"Evaluating V2 MC across ALL {len(files)} available historical ground truth files...")
    t0 = time.time()
    
    for f in files:
        basename = os.path.basename(f)
        rn = int(basename.split('_')[0].replace('round', ''))
        sn = int(basename.split('_')[1].replace('seed', '').replace('.json', ''))
        
        with open(f) as fh:
            d = json.load(fh)
        
        ig = np.array(d['initial_grid'], dtype=np.int32)
        gt = np.array(d['ground_truth'], dtype=np.float32)
        
        # 1. Gather observation queries
        obs = benchmark_experiments.simulate_observations_from_gt(gt, ig, n_viewports=3, strategy="hotspot")
        
        # 2. TTO for Regime
        best_ll = -99999999
        best_regime = 1
        
        for regime in [0, 1, 2]:
            pred = gpu_exact_mc_v2.run_gpu_mc(ig, db, regime, num_sims=500)
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
        
        # 3. Final Compute Run
        pred = gpu_exact_mc_v2.run_gpu_mc(ig, db, best_regime, num_sims=15000) # Fast evaluation
        
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
        
        wkl = gpu_exact_mc_v2.calc_wkl(snapped_prob, gt, ig)
        total_wkl += wkl
        count += 1
        print(f"R{rn} S{sn}: wKL = {wkl:.4f}")
        
    t1 = time.time()
    print(f"\nFinal GLOBAL TTO GPU MC V2 wKL: {total_wkl/count:.4f} (Time: {t1-t0:.2f}s)")

if __name__ == '__main__':
    eval_all()
