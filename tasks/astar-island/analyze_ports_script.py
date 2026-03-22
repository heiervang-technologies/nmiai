import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def analyze_ports(replay_path):
    with open(replay_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    width = data.get('width', 40)
    height = data.get('height', 40)
    
    port_class = 2
    
    prev_grid = None
    os.makedirs('port_images', exist_ok=True)
    
    for i, frame in enumerate(frames):
        grid = np.array(frame['grid'])
        
        if prev_grid is not None:
            # Find newly formed ports
            new_ports = (grid == port_class) & (prev_grid != port_class)
            
            if np.any(new_ports):
                y, x = np.where(new_ports)
                print(f"Step {frame['step']}: Port formed at {list(zip(y, x))}")
                
                # Plot the grid
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Create a custom colormap or just use a discrete one
                cax = ax.matshow(grid, cmap='tab20', vmin=0, vmax=19)
                
                # Annotate the new ports
                for py, px in zip(y, x):
                    ax.plot(px, py, 'rX', markersize=15, label='New Port' if 'New Port' not in ax.get_legend_handles_labels()[1] else "")
                
                ax.set_title(f"Step {frame['step']}")
                fig.colorbar(cax)
                plt.legend()
                
                out_path = f"port_images/step_{frame['step']}.png"
                plt.savefig(out_path)
                plt.close(fig)
                print(f"Saved {out_path}")
        
        prev_grid = grid

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_ports(sys.argv[1])
    else:
        analyze_ports('/home/me/ht/nmiai/tasks/astar-island/replays/round18_seed0_simseed1.json')
