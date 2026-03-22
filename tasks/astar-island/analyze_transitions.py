import json, glob, numpy as np
from collections import defaultdict

files = glob.glob('/home/me/ht/nmiai/tasks/astar-island/ground_truth/*.json')
transitions = defaultdict(lambda: defaultdict(int))

for f in files:
    data = json.load(open(f))
    ig = np.array(data['initial_grid'])
    gt = np.argmax(np.array(data['ground_truth']), axis=-1)
    
    for i_val in np.unique(ig):
        for g_val in np.unique(gt[ig == i_val]):
            count = np.sum((ig == i_val) & (gt == g_val))
            transitions[int(i_val)][int(g_val)] += int(count)

for i_val, g_counts in sorted(transitions.items()):
    print(f"IG {i_val}:")
    for g_val, count in sorted(g_counts.items()):
        print(f"  -> GT {g_val}: {count}")
