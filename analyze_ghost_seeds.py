import json
import matplotlib.pyplot as plt
import numpy as np
import os

filepath = "/home/me/ht/nmiai/tasks/astar-island/replays/dense_training/round17_seed0_sim1.json"

with open(filepath, 'r') as f:
    data = json.load(f)

width = data['width']
height = data['height']
frames = data['frames']

def get_neighbors(r, c, grid):
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                neighbors.append(grid[nr][nc])
    return neighbors

def plot_window(prev_grid, new_grid, r, c, step, window_size=15):
    half = window_size // 2
    r_start = max(0, r - half)
    r_end = min(height, r + half + 1)
    c_start = max(0, c - half)
    c_end = min(width, c + half + 1)

    prev_win = np.array(prev_grid)[r_start:r_end, c_start:c_end]
    new_win = np.array(new_grid)[r_start:r_end, c_start:c_end]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 1: red, 4: green, 10: blue, other: gray
    cmap_dict = {
        0: [200, 200, 200],  # Empty/Ground
        1: [255, 0, 0],      # Settlement
        2: [200, 150, 50],   # Maybe dirt/sand?
        3: [100, 200, 100],  # Light forest?
        4: [0, 255, 0],      # Forest
        10: [0, 0, 255],     # Ocean
    }
    
    def to_rgb(win):
        rgb = np.zeros((win.shape[0], win.shape[1], 3), dtype=np.uint8)
        for i in range(win.shape[0]):
            for j in range(win.shape[1]):
                val = win[i, j]
                rgb[i, j] = cmap_dict.get(val, [100, 100, 100])
        return rgb
        
    axes[0].imshow(to_rgb(prev_win))
    axes[0].set_title(f"Step {step-1}")
    
    axes[1].imshow(to_rgb(new_win))
    axes[1].set_title(f"Step {step} (Ghost Seed at center)")
    
    # draw a circle at the ghost seed location in the new window
    center_r_win = r - r_start
    center_c_win = c - c_start
    circle = plt.Circle((center_c_win, center_r_win), 0.5, color='yellow', fill=False, linewidth=2)
    axes[1].add_patch(circle)

    plt.tight_layout()
    plt.savefig(f"ghost_seed_step_{step}_{r}_{c}.png")
    plt.close()

ghost_seeds_found = 0

for i in range(1, len(frames)):
    prev_frame = frames[i-1]
    curr_frame = frames[i]
    step = curr_frame['step']
    
    prev_grid = prev_frame['grid']
    curr_grid = curr_frame['grid']
    
    for r in range(height):
        for c in range(width):
            if curr_grid[r][c] == 1 and prev_grid[r][c] != 1:
                # new settlement
                neighbors = get_neighbors(r, c, prev_grid)
                if 1 not in neighbors:
                    print(f"Ghost seed found at step {step}, position ({r}, {c})")
                    dist_to_nearest_1 = 999
                    nearest_1 = None
                    for rr in range(height):
                        for cc in range(width):
                            if prev_grid[rr][cc] == 1:
                                dist = max(abs(rr - r), abs(cc - c))
                                if dist < dist_to_nearest_1:
                                    dist_to_nearest_1 = dist
                                    nearest_1 = (rr, cc)
                    
                    print(f"  Nearest settlement at dist {dist_to_nearest_1}: {nearest_1}")
                    plot_window(prev_grid, curr_grid, r, c, step)
                    ghost_seeds_found += 1

print(f"Total Ghost Seeds found: {ghost_seeds_found}")