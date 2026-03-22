import json
import glob
import numpy as np
from scipy.ndimage import convolve
import torch
import os
import gc

def get_regime(final_grid):
    civ_count = np.sum((final_grid == 1) | (final_grid == 2))
    if civ_count < 30: return 0
    if civ_count > 150: return 2
    return 1

def build_sparse_tensor():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f]
    
    print(f"Building sparse lookup tensor from {len(files)} replays...")

    k3 = np.ones((3,3), dtype=np.int32)
    k3[1,1] = 0
    k7 = np.ones((7,7), dtype=np.int32)
    k7[3,3] = 0
    
    # We will use a dictionary of tensors to save memory
    # Key: (phase, regime, center)
    # Value: Dict of (civ3, civ7, ocean3, forest3, ruin3, port3) -> counts array
    transitions = {}
    
    for f_idx, f in enumerate(files):
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
                    nxt = curr[y,x]
                    
                    # Group by main factors to reduce top-level keys
                    main_key = (phase, regime, center)
                    if main_key not in transitions:
                        transitions[main_key] = {}
                        
                    c3 = n_civ3[y,x]
                    c7 = n_civ7[y,x]
                    o3 = n_ocean3[y,x]
                    f3 = n_forest3[y,x]
                    r3 = n_ruin3[y,x]
                    p3 = n_port3[y,x]
                    
                    sub_key = (c3, c7, o3, f3, r3, p3)
                    
                    if sub_key not in transitions[main_key]:
                        transitions[main_key][sub_key] = np.zeros(12, dtype=np.float32)
                    transitions[main_key][sub_key][nxt] += 1
                    
        if (f_idx + 1) % 10 == 0:
            print(f"Processed {f_idx + 1} / {len(files)} files...")
            
    print("Normalizing to probabilities and saving...")
    
    # We serialize as a dict to avoid massive zeros in a dense tensor
    for main_key in transitions:
        for sub_key in transitions[main_key]:
            counts = transitions[main_key][sub_key]
            total = counts.sum()
            if total > 0:
                transitions[main_key][sub_key] = counts / total
                
    torch.save(transitions, 'tasks/astar-island/sparse_lookup_v2.pt')
    print("Saved to sparse_lookup_v2.pt")

if __name__ == '__main__':
    build_sparse_tensor()
