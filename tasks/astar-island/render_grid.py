import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

# Load the JSON
with open('replays/dense_training/round17_seed0_sim1.json', 'r') as f:
    data = json.load(f)

# Create output dir
os.makedirs('analysis_frames', exist_ok=True)

# Define colormap
colors = ['white', 'blue', 'green', 'yellow', 'brown', 'purple', 'gray', 'pink', 'orange', 'cyan', 'black', 'lightblue', 'lightgreen']
cmap = mcolors.ListedColormap(colors[:12]) # Ensure we cover enough terrain types

for frame in data['frames']:
    step = frame['step']
    # Let's render first 20 steps
    if step <= 20:
        grid = np.array(frame['grid'])
        settlements = frame['settlements']
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='tab20', interpolation='nearest', vmin=np.min(grid)-0.5, vmax=np.max(grid)+0.5)
        plt.colorbar(ticks=np.arange(np.min(grid), np.max(grid)+1))
        
        # Plot settlements
        xs = [s['x'] for s in settlements if s['alive']]
        ys = [s['y'] for s in settlements if s['alive']]
        pops = [s['population'] * 20 for s in settlements if s['alive']] 
        
        plt.scatter(xs, ys, s=pops, c='red', edgecolor='white', alpha=0.7)
        plt.title(f'Step {step} - Settlements: {len(xs)}')
        
        plt.savefig(f'analysis_frames/step_{step:03d}.png')
        plt.close()
        
        print(f"Generated step {step} with {len(xs)} settlements")
