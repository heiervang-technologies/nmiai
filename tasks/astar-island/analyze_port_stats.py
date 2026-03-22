import json
import numpy as np
from scipy.ndimage import label
import sys

def analyze(replay_path):
    with open(replay_path, 'r') as f:
        data = json.load(f)
        
    frames = data['frames']
    port_class = 2
    settlement_class = 1 # guessing 1 is settlement
    
    prev_grid = None
    min_dist = 999
    
    all_ports_ever = set()
    
    for frame in frames:
        grid = np.array(frame['grid'])
        
        # classes present
        classes = np.unique(grid)
        
        # calculate distance between any two ports in the current frame
        ports = np.argwhere(grid == port_class)
        for i in range(len(ports)):
            for j in range(i+1, len(ports)):
                dist = np.max(np.abs(ports[i] - ports[j])) # Chebyshev
                if dist < min_dist:
                    min_dist = dist
        
        if prev_grid is not None:
            new_ports = (grid == port_class) & (prev_grid != port_class)
            y, x = np.where(new_ports)
            for py, px in zip(y, x):
                # Look at 5x5 neighborhood in prev_grid
                y0, y1 = max(0, py-2), min(grid.shape[0], py+3)
                x0, x1 = max(0, px-2), min(grid.shape[1], px+3)
                neigh = prev_grid[y0:y1, x0:x1]
                unique, counts = np.unique(neigh, return_counts=True)
                counts_dict = dict(zip(unique, counts))
                
                # compute connected component size of settlements (class 1)
                settlements = (prev_grid == 1).astype(int)
                labeled_array, num_features = label(settlements)
                # check neighbors of (py, px) for settlements
                adj_settlement_labels = set()
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0: continue
                        ny, nx = py+dy, px+dx
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            if settlements[ny, nx]:
                                adj_settlement_labels.add(labeled_array[ny, nx])
                
                cluster_size = 0
                for lab in adj_settlement_labels:
                    cluster_size += np.sum(labeled_array == lab)
                
                print(f"Port at ({py},{px}): min port dist={min_dist}, classes in 5x5={counts_dict}, attached settlement cluster size={cluster_size}")
                
        prev_grid = grid

    print(f"Overall min Chebyshev distance between ports: {min_dist}")

analyze('/home/me/ht/nmiai/tasks/astar-island/replays/round18_seed0_simseed1.json')
