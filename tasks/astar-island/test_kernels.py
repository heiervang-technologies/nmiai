import json
import glob
import numpy as np
from scipy.ndimage import convolve
from scipy.stats import pearsonr

def test_kernel_sizes():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f]
    
    print(f"Testing kernel sizes over {len(files)} replays...")

    # We will test radius 1 (3x3), radius 2 (5x5), and radius 3 (7x7)
    k3 = np.ones((3,3), dtype=np.int32)
    k3[1,1] = 0
    
    k5 = np.ones((5,5), dtype=np.int32)
    k5[2,2] = 0
    
    k7 = np.ones((7,7), dtype=np.int32)
    k7[3,3] = 0

    all_ruin_events = [] # 1 if settlement collapsed, 0 if it survived
    civ_counts_3 = []
    civ_counts_5 = []
    civ_counts_7 = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 2: continue
        
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            
            is_civ = ((prev == 1) | (prev == 2)).astype(np.int32)
            is_settle = (prev == 1)
            
            if not np.any(is_settle): continue
                
            n3 = convolve(is_civ, k3, mode='constant')
            n5 = convolve(is_civ, k5, mode='constant')
            n7 = convolve(is_civ, k7, mode='constant')
            
            collapsed = (curr == 3)
            
            y_idx, x_idx = np.where(is_settle)
            
            # Sample to avoid exploding memory (take up to 500 per frame)
            if len(y_idx) > 500:
                indices = np.random.choice(len(y_idx), 500, replace=False)
            else:
                indices = range(len(y_idx))
                
            for idx in indices:
                y = y_idx[idx]
                x = x_idx[idx]
                
                all_ruin_events.append(1 if collapsed[y, x] else 0)
                civ_counts_3.append(n3[y, x])
                civ_counts_5.append(n5[y, x])
                civ_counts_7.append(n7[y, x])

    r3, _ = pearsonr(civ_counts_3, all_ruin_events)
    r5, _ = pearsonr(civ_counts_5, all_ruin_events)
    r7, _ = pearsonr(civ_counts_7, all_ruin_events)

    print("\n--- Correlation with Settlement Collapse (Settle -> Ruin) ---")
    print(f"Radius 1 (3x3 kernel): r = {r3:.4f}")
    print(f"Radius 2 (5x5 kernel): r = {r5:.4f}")
    print(f"Radius 3 (7x7 kernel): r = {r7:.4f}")

if __name__ == '__main__':
    test_kernel_sizes()