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

def dump_entropy():
    print("Loading sparse lookup dictionary...")
    transitions = torch.load('tasks/astar-island/sparse_lookup_v2.pt', weights_only=False)
    
    # Calculate entropy for all states to find the highest ones
    state_entropies = {}
    for main_key, sub_dict in transitions.items():
        for sub_key, p_arr in sub_dict.items():
            # calculate entropy
            p_safe = np.maximum(p_arr, 1e-9)
            entropy = -np.sum(p_safe * np.log2(p_safe))
            if entropy > 0.8: # high entropy threshold
                state_entropies[(main_key, sub_key)] = (entropy, p_arr)
                
    print(f"Found {len(state_entropies)} highly stochastic state configurations.")
    
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f][:5] # just check a few to find examples
    
    k3 = np.ones((3,3), dtype=np.int32)
    k3[1,1] = 0
    k7 = np.ones((7,7), dtype=np.int32)
    k7[3,3] = 0
    
    class_map = {0: 'Empty', 1: 'Settle', 2: 'Port', 3: 'Ruin', 4: 'Forest', 5: 'Mount', 10: 'Ocean', 11: 'Plains'}
    
    targets_found = 0
    out_lines = ["# High Entropy Event Targets for Qualitative Analysis\n"]
    out_lines.append("Review these exact coordinates in the replay images to see if visual context explains the divergent outcomes.\n\n")

    for f in files:
        if targets_found > 20: break
        
        basename = os.path.basename(f)
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 51: continue
        
        final_grid = np.array(frames[-1]['grid'])
        regime = get_regime(final_grid)
        
        for i in range(1, len(frames)):
            if targets_found > 20: break
            
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
                        entropy, p_arr = state_entropies[(main_key, sub_key)]
                        nxt = curr[y,x]
                        
                        reg_str = ["Harsh", "Moderate", "Prosperous"][regime]
                        out_lines.append(f"### Target: {basename} | Step {i} -> {i+1} | Coord: (y={y}, x={x})")
                        out_lines.append(f"- **State:** {class_map.get(center, center)} (Phase {phase}, {reg_str})")
                        out_lines.append(f"- **Neighbors:** civ3={sub_key[0]}, civ7={sub_key[1]}, oc3={sub_key[2]}, for3={sub_key[3]}, ruin3={sub_key[4]}, port3={sub_key[5]}")
                        out_lines.append(f"- **Entropy:** {entropy:.2f} bits")
                        
                        out_str = "- **Distribution:** "
                        for c in range(12):
                            if p_arr[c] > 0.05:
                                out_str += f"{class_map.get(c, c)}: {p_arr[c]*100:.1f}% | "
                        out_lines.append(out_str)
                        out_lines.append(f"- **Actual Outcome:** {class_map.get(nxt, nxt)}\n")
                        
                        targets_found += 1
                        if targets_found > 20: break

    with open('tasks/astar-island/high_entropy_targets.md', 'w') as f:
        f.write('\n'.join(out_lines))
    print("Dumped targets to tasks/astar-island/high_entropy_targets.md")

if __name__ == '__main__':
    dump_entropy()
