import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def render_probes():
    # specifically target round22
    latest_round = 'tasks/astar-island/logs/round22'
    
    files = sorted(glob.glob(os.path.join(latest_round, 'observations_seed*.json')))
    if not files:
        print(f"No observations found in {latest_round}")
        return
        
    fig, axes = plt.subplots(1, len(files), figsize=(5*len(files), 5))
    if len(files) == 1:
        axes = [axes]
        
    # Colormap mimicking the terrain
    colors = {
        0: '#000000', # Empty
        1: '#FF0000', # Settle
        2: '#00FFFF', # Port
        3: '#888888', # Ruin
        4: '#00FF00', # Forest
        5: '#FFFFFF', # Mount
        10: '#0000FF', # Ocean
        11: '#FFFF00'  # Plains
    }
    
    for i, f in enumerate(files):
        with open(f) as fh:
            obs = json.load(fh)
            
        # Draw a blank 40x40 grid
        grid = np.zeros((40, 40, 3))
        
        for ob in obs:
            vx, vy = ob['viewport_x'], ob['viewport_y']
            vg = ob['grid']
            for y in range(len(vg)):
                for x in range(len(vg[0])):
                    val = vg[y][x]
                    hex_color = colors.get(val, '#000000')
                    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4))
                    grid[vy+y, vx+x] = rgb
                    
        axes[i].imshow(grid)
        axes[i].set_title(f"Seed {i} Probes")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('tasks/astar-island/probes_vision.png')
    print("Saved to tasks/astar-island/probes_vision.png")

if __name__ == '__main__':
    render_probes()
