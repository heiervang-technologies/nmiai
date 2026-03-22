import json
import glob
import numpy as np

def find_thresholds():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f]
    
    step4_counts = {'Harsh': [], 'Moderate': [], 'Prosperous': []}
    
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 51: continue
        
        final_grid = np.array(frames[-1]['grid'])
        civ_count_final = np.sum((final_grid == 1) | (final_grid == 2))
        
        if civ_count_final < 30: regime = 'Harsh'
        elif civ_count_final > 150: regime = 'Prosperous'
        else: regime = 'Moderate'
        
        step4_grid = np.array(frames[4]['grid'])
        civ_count_4 = np.sum((step4_grid == 1) | (step4_grid == 2))
        
        step4_counts[regime].append(civ_count_4)
        
    for r in ['Harsh', 'Moderate', 'Prosperous']:
        if step4_counts[r]:
            print(f"{r}: min={min(step4_counts[r])}, max={max(step4_counts[r])}, mean={np.mean(step4_counts[r]):.1f}, samples={len(step4_counts[r])}")

if __name__ == '__main__':
    find_thresholds()
