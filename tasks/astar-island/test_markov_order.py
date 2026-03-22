import torch
import json
import glob
import numpy as np
from scipy.ndimage import convolve
import os

def test_markov():
    print("Testing if CA is a 2nd-order Markov process...")
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f][:30]
    
    k3 = np.ones((3,3), dtype=np.int32)
    k3[1,1] = 0
    k7 = np.ones((7,7), dtype=np.int32)
    k7[3,3] = 0
    
    # We want to see if given the exact same 9-dimensional state at t, 
    # the probability of state t+1 changes depending on state at t-1.
    
    # state_counts: (9-dim key) -> { (center state at t-1) -> counts }
    transitions = {}
    
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 51: continue
        
        for i in range(2, len(frames)):
            prev_prev = np.array(frames[i-2]['grid'])
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
                    center_prev_prev = prev_prev[y,x]
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
                            
                    state_key = (phase, center, c3, c7, o3, f3, r3, p3)
                    if state_key not in transitions:
                        transitions[state_key] = {}
                    
                    if center_prev_prev not in transitions[state_key]:
                        transitions[state_key][center_prev_prev] = np.zeros(12, dtype=np.int32)
                    
                    transitions[state_key][center_prev_prev][nxt] += 1

    # Check if distributions differ significantly based on prev_prev state
    divergence_count = 0
    total_stochastic_keys = 0
    
    for key, prev_dict in transitions.items():
        if len(prev_dict) > 1:
            # We have multiple t-1 states leading to this t state
            # Let's see if the output distributions are different
            dist_list = []
            for prev_state, counts in prev_dict.items():
                if counts.sum() > 20: # need minimum sample size
                    dist_list.append(counts / counts.sum())
                    
            if len(dist_list) > 1:
                total_stochastic_keys += 1
                # Check max difference between distributions
                max_diff = 0
                for i in range(len(dist_list)):
                    for j in range(i+1, len(dist_list)):
                        diff = np.max(np.abs(dist_list[i] - dist_list[j]))
                        if diff > max_diff:
                            max_diff = diff
                if max_diff > 0.2: # more than 20% difference in probability
                    divergence_count += 1
                    
    print(f"Total stochastic states checked for 2nd order Markov property: {total_stochastic_keys}")
    print(f"Number of states where t-1 state significantly changes t+1 probabilities: {divergence_count}")

if __name__ == '__main__':
    test_markov()
