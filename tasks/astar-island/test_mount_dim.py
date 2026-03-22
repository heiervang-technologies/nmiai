import json
import glob
import numpy as np
import pickle
from scipy.ndimage import convolve

def test_mount_dim():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f][:30]
    
    k3 = np.ones((3,3), dtype=np.int32)
    k3[1,1] = 0
    k7 = np.ones((7,7), dtype=np.int32)
    k7[3,3] = 0
    
    transitions = {}
    
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 51: continue
        
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
            
            is_mount = (prev == 5).astype(np.int32)
            n_mount3 = convolve(is_mount, k3, mode='constant')
            
            for y in range(40):
                for x in range(40):
                    center = prev[y,x]
                    nxt = curr[y,x]
                    
                    if center == nxt and center in (0, 10, 11) and n_civ3[y,x]==0 and n_civ7[y,x]==0 and n_ruin3[y,x]==0 and n_forest3[y,x]==0 and n_port3[y,x]==0 and n_mount3[y,x]==0:
                        if np.random.rand() > 0.05:
                            continue
                    
                    key = (phase, center, n_civ3[y,x], n_civ7[y,x], n_ocean3[y,x], n_forest3[y,x], n_ruin3[y,x], n_port3[y,x], n_mount3[y,x])
                    if key not in transitions:
                        transitions[key] = np.zeros(12, dtype=np.int32)
                    transitions[key][nxt] += 1
                    
    total_samples = 0
    deterministic_samples = 0
    
    for k, counts in transitions.items():
        s = counts.sum()
        total_samples += s
        if np.max(counts) == s:
            deterministic_samples += s
            
    print(f"Deterministic samples with mount3: {deterministic_samples/total_samples*100:.2f}%")

if __name__ == '__main__':
    test_mount_dim()
