import json
import glob
import numpy as np
import torch
import os
import sys
from scipy.ndimage import distance_transform_edt

sys.path.append(os.path.dirname(__file__))
from gpu_exact_mc import run_gpu_mc, calc_wkl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_structural_zeros(pred, ig):
    pred = np.maximum(pred, 1e-8)
    pred[ig == 5] = [0, 0, 0, 0, 0, 1]
    pred[ig == 10] = [1, 0, 0, 0, 0, 0]
    mask = (ig == 1) | (ig == 2)
    dist = distance_transform_edt(~mask) if mask.any() else np.full(ig.shape, 100, dtype=np.float64)
    pred[(dist > 12) & ((ig == 1) | (ig == 2) | (ig == 0) | (ig == 11))] = [1, 0, 0, 0, 0, 0]
    pred[(dist > 12) & (ig == 4)] = [0, 0, 0, 0, 1, 0]
    return pred

data = torch.load('tasks/astar-island/round_tensors/all_rounds.pt', weights_only=False)
all_tensors = data['tensors'].to(device)
round_nums = data['round_nums']

results = {}
files = glob.glob('tasks/astar-island/ground_truth/*.json')

for ti, rn in enumerate(round_nums):
    tensor = all_tensors[ti]
    round_files = [f for f in files if f"round{rn}_" in os.path.basename(f)]
    if not round_files:
        continue
    
    total_wkl = 0
    for f in round_files:
        with open(f) as fh:
            d = json.load(fh)
        ig = np.array(d['initial_grid'], dtype=np.int32)
        gt = np.array(d['ground_truth'], dtype=np.float32)
        
        pred = run_gpu_mc(ig, tensor, num_sims=1000)
        pred = apply_structural_zeros(pred, ig)
        wkl = calc_wkl(pred, gt, ig)
        total_wkl += wkl
        
    mean_wkl = total_wkl / len(round_files)
    results[rn] = float(mean_wkl)
    print(f"R{rn}: {mean_wkl:.4f}")

with open('tasks/astar-island/historical_wkl.json', 'w') as f:
    json.dump(results, f, indent=2)
