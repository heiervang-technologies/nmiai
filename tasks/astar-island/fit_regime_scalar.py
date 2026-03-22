import json
import glob
import numpy as np
import os

def get_stats():
    files = glob.glob('tasks/astar-island/replays/*.json')
    round_stats = {}
    
    global_new_civ = 0
    global_non_civ = 0

    print("Parsing replays to calculate true transition scalars...")
    for f in files:
        rnd_str = os.path.basename(f).split('_')[0].replace('round', '')
        if not rnd_str.isdigit(): continue
        rnd = int(rnd_str)
        
        with open(f) as fh:
            try:
                data = json.load(fh)
            except:
                continue
                
        frames = data.get('frames', [])
        if len(frames) < 50: continue
        
        final_grid = np.array(frames[-1]['grid'])
        civ_count = np.sum((final_grid == 1) | (final_grid == 2))
        
        # Calculate transition probability to civ
        new_civ_transitions = 0
        non_civ_opportunities = 0
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            
            # Non-civ in prev (excluding Mountains and Oceans which can't be settled)
            prev_non_civ = ~((prev == 1) | (prev == 2) | (prev == 5) | (prev == 10))
            non_civ_opportunities += np.sum(prev_non_civ)
            
            # Became civ in curr
            became_civ = prev_non_civ & ((curr == 1) | (curr == 2))
            new_civ_transitions += np.sum(became_civ)
            
        if rnd not in round_stats:
            round_stats[rnd] = {'civ_counts': [], 'new_civ': 0, 'opps': 0}
            
        round_stats[rnd]['civ_counts'].append(civ_count)
        round_stats[rnd]['new_civ'] += new_civ_transitions
        round_stats[rnd]['opps'] += non_civ_opportunities
        
        global_new_civ += new_civ_transitions
        global_non_civ += non_civ_opportunities
        
    global_prob = global_new_civ / max(1, global_non_civ)
    
    X = []
    Y = []
    
    print(f"\nGlobal base settlement prob: {global_prob:.5f}")
    print("-" * 50)
    print(f"{'Round':>5} | {'Mean Civ':>8} | {'Scalar':>8}")
    print("-" * 50)
    
    for rnd in sorted(round_stats.keys()):
        stats = round_stats[rnd]
        mean_civ = np.mean(stats['civ_counts'])
        round_prob = stats['new_civ'] / max(1, stats['opps'])
        scalar = round_prob / global_prob
        
        X.append(mean_civ)
        Y.append(scalar)
        print(f"{rnd:5} | {mean_civ:8.1f} | {scalar:8.4f}")
        
    # Fit polynomial (degree 2)
    p = np.polyfit(X, Y, 2)
    print(f"\nFitted polynomial coefficients (degree 2): {p}")
    
    # Save the model
    with open('tasks/astar-island/regime_model.json', 'w') as f:
        json.dump({'poly_coeffs': p.tolist(), 'global_prob': global_prob}, f, indent=2)
        
    print("Saved polynomial mapping to regime_model.json")

if __name__ == '__main__':
    get_stats()
