import json
import glob

files = glob.glob('replays/dense_training/round17_seed0_sim*.json')
for f in files[:5]:
    with open(f, 'r') as file:
        data = json.load(file)
        
    counts = []
    for frame in data['frames']:
        alive = len([s for s in frame['settlements'] if s['alive']])
        counts.append(alive)
        
    drops = []
    for i in range(1, len(counts)):
        if counts[i-1] - counts[i] > 10:
            drops.append((i, counts[i-1] - counts[i]))
    print(f"{f}: drops at {drops}")
