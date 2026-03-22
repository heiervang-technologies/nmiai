import json
import glob
import numpy as np
from scipy.ndimage import convolve

def get_regime(final_grid):
    civ_count = np.sum((final_grid == 1) | (final_grid == 2))
    if civ_count < 30: return 0
    if civ_count > 150: return 2
    return 1

def extract_tensor():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f][:30]
    
    print(f"Testing 4-Phase determinism with ruin3 and port3 on {len(files)} replays...")

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
        
        final_grid = np.array(frames[-1]['grid'])
        regime = get_regime(final_grid)
        
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            phase = i % 4  # Back to mod 4
            
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
                    
                    c3 = n_civ3[y,x]
                    c7 = n_civ7[y,x]
                    o3 = n_ocean3[y,x]
                    f3 = n_forest3[y,x]
                    r3 = n_ruin3[y,x]
                    p3 = n_port3[y,x]
                    
                    # Optimization
                    if center == nxt and center in (0, 10, 11) and c3==0 and c7==0 and r3==0 and f3==0 and p3==0:
                        if np.random.rand() > 0.05:
                            continue
                    
                    key = (phase, regime, center, c3, c7, o3, f3, r3, p3)
                    if key not in transitions:
                        transitions[key] = np.zeros(12, dtype=np.int32)
                    transitions[key][nxt] += 1
                    
    total_samples = 0
    deterministic_samples = 0
    stochastic_samples = 0
    
    for k, counts in transitions.items():
        s = counts.sum()
        total_samples += s
        if np.max(counts) == s:
            deterministic_samples += s
        else:
            stochastic_samples += s
            
    print(f"Total samples (sampled): {total_samples}")
    print(f"Deterministic samples: {deterministic_samples} ({deterministic_samples/total_samples*100:.2f}%)")
    print(f"Stochastic samples: {stochastic_samples} ({stochastic_samples/total_samples*100:.2f}%)")

if __name__ == '__main__':
    extract_tensor()
