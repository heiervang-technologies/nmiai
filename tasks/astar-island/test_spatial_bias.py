import torch
import json
import glob
import numpy as np
from scipy.ndimage import convolve
import os

def get_regime(final_grid):
    civ_count = np.sum((final_grid == 1) | (final_grid == 2))
    if civ_count < 30: return 0
    if civ_count > 150: return 2
    return 1

def analyze_spatial_bias():
    print("Loading sparse lookup dictionary...")
    transitions = torch.load('tasks/astar-island/sparse_lookup_v2.pt', weights_only=False)
    
    state_entropies = {}
    for main_key, sub_dict in transitions.items():
        for sub_key, p_arr in sub_dict.items():
            p_safe = np.maximum(p_arr, 1e-9)
            entropy = -np.sum(p_safe * np.log2(p_safe))
            if entropy > 0.8:
                state_entropies[(main_key, sub_key)] = True
                
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f][:20]
    
    k3 = np.ones((3,3), dtype=np.int32)
    k3[1,1] = 0
    k7 = np.ones((7,7), dtype=np.int32)
    k7[3,3] = 0
    
    y_coords = []
    x_coords = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 51: continue
        
        final_grid = np.array(frames[-1]['grid'])
        regime = get_regime(final_grid)
        
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            phase = i % 4
            
            is_civ = ((prev == 1) | (prev == 2)).astype(np.int32)
            n_civ3 = convolve(is_civ, k3, mode='constant')
            n_civ7 = convolve(is_civ, k7, mode='constant')
            n_civ7 = np.clip(n_civ7, 0, 25)
            
            is_ocean = (prev == 10).astype(np.int32)
            n_ocean3 = convolve(is_ocean, k3, mode='constant')
            
            is_forest = (prev == 4).astype(np.int32)
            n_forest3 = convolve(is_forest, k3, mode='constant')
            
            is_ruin = (prev == 3).astype(np.int32)
            n_ruin3 = convolve(is_ruin, k3, mode='constant')
            
            is_port = (prev == 2).astype(np.int32)
            n_port3 = convolve(is_port, k3, mode='constant')
            
            for y in range(40):
                for x in range(40):
                    center = prev[y,x]
                    main_key = (phase, regime, center)
                    sub_key = (n_civ3[y,x], n_civ7[y,x], n_ocean3[y,x], n_forest3[y,x], n_ruin3[y,x], n_port3[y,x])
                    
                    if (main_key, sub_key) in state_entropies:
                        y_coords.append(y)
                        x_coords.append(x)

    print(f"Analyzed {len(y_coords)} high entropy events.")
    print(f"Y coord mean: {np.mean(y_coords):.2f}, std: {np.std(y_coords):.2f}")
    print(f"X coord mean: {np.mean(x_coords):.2f}, std: {np.std(x_coords):.2f}")
    
    # Check for quadrant bias
    q1 = sum(1 for y, x in zip(y_coords, x_coords) if y < 20 and x < 20)
    q2 = sum(1 for y, x in zip(y_coords, x_coords) if y < 20 and x >= 20)
    q3 = sum(1 for y, x in zip(y_coords, x_coords) if y >= 20 and x < 20)
    q4 = sum(1 for y, x in zip(y_coords, x_coords) if y >= 20 and x >= 20)
    
    print(f"Quadrants (TL, TR, BL, BR): {q1}, {q2}, {q3}, {q4}")

if __name__ == '__main__':
    analyze_spatial_bias()
