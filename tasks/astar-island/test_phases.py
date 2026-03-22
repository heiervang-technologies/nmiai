import json
import glob
import numpy as np

def test_phases():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f][:10]
    
    mod4_changes = {0:0, 1:0, 2:0, 3:0}
    mod5_changes = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            diff = np.sum(prev != curr)
            mod4_changes[i % 4] += diff
            mod5_changes[i % 5] += diff
            
    print("Mod 4 total changes by phase:", mod4_changes)
    print("Mod 5 total changes by phase:", mod5_changes)

if __name__ == '__main__':
    test_phases()
