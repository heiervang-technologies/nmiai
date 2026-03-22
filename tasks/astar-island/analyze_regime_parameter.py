import json
import glob
import numpy as np
import os

def analyze_regime():
    files = glob.glob('tasks/astar-island/replays/*.json')
    round_stats = {}
    
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
        
        if rnd not in round_stats:
            round_stats[rnd] = []
        round_stats[rnd].append(civ_count)
        
    print(f"{'Round':>5} | {'Replays':>7} | {'Mean Civ':>8} | {'Std Dev':>7}")
    print("-" * 37)
    
    for rnd in sorted(round_stats.keys()):
        counts = round_stats[rnd]
        print(f"{rnd:5} | {len(counts):7} | {np.mean(counts):8.1f} | {np.std(counts):7.1f}")

if __name__ == '__main__':
    analyze_regime()
