import json
import glob
import numpy as np
from collections import defaultdict
from scipy.ndimage import convolve

def get_regime(final_grid):
    civ_count = np.sum((final_grid == 1) | (final_grid == 2))
    if civ_count < 30: return 'Harsh'
    if civ_count > 150: return 'Prosperous'
    return 'Moderate'

def build_tables():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f]
    
    # Tables: table[regime][phase][state][civ_neighbors][next_state] = probability
    # We'll simplify to just civ_neighbors for the baseline to avoid sparsity issues
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))
    
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)
    
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 51: continue
        
        regime = get_regime(np.array(frames[-1]['grid']))
        
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            phase = i % 4
            
            is_civ = ((prev == 1) | (prev == 2)).astype(np.int32)
            n_civ = convolve(is_civ, kernel, mode='constant')
            
            # Sample all cells to build the full denominator
            for y in range(40):
                for x in range(40):
                    old_v = int(prev[y, x])
                    new_v = int(curr[y, x])
                    civ_c = int(n_civ[y, x])
                    
                    if old_v in (0, 5, 10): continue # Skip empty, mountain, ocean (static or mostly static)
                    
                    counts[regime][phase][old_v][civ_c][new_v] += 1
                    
    # Convert counts to probabilities
    probs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
    for r in counts:
        for p in counts[r]:
            for st in counts[r][p]:
                for cv in counts[r][p][st]:
                    total = sum(counts[r][p][st][cv].values())
                    for nxt in counts[r][p][st][cv]:
                        probs[r][p][st][cv][nxt] = counts[r][p][st][cv][nxt] / total
                        
    with open('tasks/astar-island/mc_transition_tables.json', 'w') as f:
        json.dump(probs, f, indent=2)
    print("Built and saved transition tables to mc_transition_tables.json")

if __name__ == '__main__':
    build_tables()
