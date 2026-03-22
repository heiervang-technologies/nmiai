import json
import numpy as np

def analyze(replay_path):
    with open(replay_path, 'r') as f:
        data = json.load(f)
        
    frames = data['frames']
    port_class = 2
    
    prev_grid = None
    
    for frame in frames:
        grid = np.array(frame['grid'])
        
        if prev_grid is not None:
            new_ports = (grid == port_class) & (prev_grid != port_class)
            y, x = np.where(new_ports)
            for py, px in zip(y, x):
                # Look at 3x3 neighborhood in prev_grid
                y0, y1 = max(0, py-1), min(grid.shape[0], py+2)
                x0, x1 = max(0, px-1), min(grid.shape[1], px+2)
                neigh = prev_grid[y0:y1, x0:x1]
                
                # Check 3x3 neighborhood in current_grid as well?
                neigh_curr = grid[y0:y1, x0:x1]
                
                print(f"Step {frame['step']}, Port at ({py},{px}):")
                print("Prev 3x3:")
                print(neigh)
                #print("Curr 3x3:")
                #print(neigh_curr)
                print("---")
                
        prev_grid = grid

analyze('/home/me/ht/nmiai/tasks/astar-island/replays/round18_seed0_simseed1.json')
