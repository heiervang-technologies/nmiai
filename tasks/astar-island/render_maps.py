import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def render_maps():
    files = sorted(glob.glob('/home/me/ht/nmiai/tasks/astar-island/ground_truth/*.json'))
    # Pick a few distinct ones
    files_to_render = [
        [f for f in files if 'round1_' in f][0],
        [f for f in files if 'round5_' in f][0],
        [f for f in files if 'round10_' in f][0],
        [f for f in files if 'round15_' in f][0]
    ]

    # Color mapping
    # 0: Plains/Empty (Tan)
    # 1: Settlement (Red)
    # 2: Port (Purple)
    # 3: Farm/Road/Unknown (Orange)
    # 4: Forest (Green)
    # 5: Mountain (Gray)
    # 10: Ocean (Blue)
    
    color_dict = {
        0: (210/255, 180/255, 140/255), # Tan
        1: (255/255, 0/255, 0/255),     # Red
        2: (128/255, 0/255, 128/255),   # Purple
        3: (255/255, 165/255, 0/255),   # Orange
        4: (34/255, 139/255, 34/255),   # Green
        5: (128/255, 128/255, 128/255), # Gray
        10: (0/255, 0/255, 255/255),    # Blue
        11: (210/255, 180/255, 140/255), # Tan
    }
    
    for idx, f in enumerate(files_to_render):
        with open(f) as fh:
            data = json.load(fh)
            
        ig = np.array(data['initial_grid'])
        gt = np.array(data['ground_truth'])
        gt_argmax = np.argmax(gt, axis=-1)
        
        # Create RGB images
        h, w = ig.shape
        ig_rgb = np.zeros((h, w, 3))
        gt_rgb = np.zeros((h, w, 3))
        
        for val, color in color_dict.items():
            ig_rgb[ig == val] = color
            gt_rgb[gt_argmax == val] = color
            
        # Unmapped values get black
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(ig_rgb)
        axes[0].set_title(f'Initial Grid ({os.path.basename(f)})')
        axes[0].axis('off')
        
        axes[1].imshow(gt_rgb)
        axes[1].set_title(f'Ground Truth Argmax ({os.path.basename(f)})')
        axes[1].axis('off')
        
        out_name = f'/home/me/ht/nmiai/tasks/astar-island/render_{idx}.png'
        plt.tight_layout()
        plt.savefig(out_name)
        plt.close()
        print(f"Saved {out_name}")

if __name__ == "__main__":
    render_maps()
