import json
import glob
import numpy as np
import os
from collections import defaultdict

def analyze_replays():
    files = glob.glob('tasks/astar-island/replays/*.json')
    # Pick a few distinct rounds (e.g. prosperous vs harsh)
    # Just grab 5 random files to start
    files = files[:5]
    
    print(f"Analyzing {len(files)} replay files...")
    
    changes_per_step = defaultdict(list)
    types_of_changes_per_step = defaultdict(lambda: defaultdict(int))
    first_change_dist = []
    
    # 0: Empty, 1: Settlement, 2: Port, 3: Ruin, 4: Forest, 5: Mountain, 10: Ocean, 11: Plains
    # Let's map these to simple names for analysis
    class_map = {0: 'Empty', 1: 'Settlement', 2: 'Port', 3: 'Ruin', 4: 'Forest', 5: 'Mountain', 10: 'Ocean', 11: 'Plains'}
    
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            
        frames = data['frames']
        if not frames:
            continue
            
        for i in range(1, len(frames)):
            prev_grid = np.array(frames[i-1]['grid'])
            curr_grid = np.array(frames[i]['grid'])
            
            diff_mask = prev_grid != curr_grid
            num_changes = np.sum(diff_mask)
            changes_per_step[i].append(num_changes)
            
            if num_changes > 0:
                y_idx, x_idx = np.where(diff_mask)
                for y, x in zip(y_idx, x_idx):
                    old_val = prev_grid[y, x]
                    new_val = curr_grid[y, x]
                    change_str = f"{class_map.get(old_val, str(old_val))} -> {class_map.get(new_val, str(new_val))}"
                    types_of_changes_per_step[i][change_str] += 1
                    
        # Find which cells change first
        initial_grid = np.array(frames[0]['grid'])
        civ_mask = (initial_grid == 1) | (initial_grid == 2)
        if civ_mask.any():
            from scipy.ndimage import distance_transform_edt
            dist_to_civ = distance_transform_edt(~civ_mask)
            
            for i in range(1, len(frames)):
                prev_grid = np.array(frames[i-1]['grid'])
                curr_grid = np.array(frames[i]['grid'])
                diff_mask = prev_grid != curr_grid
                if diff_mask.any():
                    dists = dist_to_civ[diff_mask]
                    for d in dists:
                        first_change_dist.append((i, d))
                    break # Only log the first time step where a change occurs for this map

    # Generate Report
    md = "# Astar Island Replay Analysis (Frame-by-Frame)\n\n"
    
    md += "## 1. Average Changes Per Step\n"
    avg_changes = {step: np.mean(counts) for step, counts in sorted(changes_per_step.items())}
    for step in range(1, 51):
        md += f"- **Step {step}:** {avg_changes.get(step, 0):.1f} cells changed\n"
        
    md += "\n## 2. Clustering & Phasing (The 5 Phases)\n"
    md += "The rules mention: *growth, conflict, trade, harsh winters, and environmental change*. Do we see this in the data?\n\n"
    
    for step in range(1, 51):
        if step in types_of_changes_per_step and len(types_of_changes_per_step[step]) > 0:
            top_changes = sorted(types_of_changes_per_step[step].items(), key=lambda x: x[1], reverse=True)[:3]
            changes_str = ", ".join([f"{c} ({cnt})" for c, cnt in top_changes])
            if sum(types_of_changes_per_step[step].values()) > 0:
                md += f"- **Step {step}:** Top changes: {changes_str}\n"

    md += "\n## 3. Which cells change first?\n"
    if first_change_dist:
        avg_first_dist = np.mean([d for _, d in first_change_dist])
        md += f"The very first cells to change state are on average **{avg_first_dist:.2f} cells away** from the initial settlements.\n"
        dists_only = [d for _, d in first_change_dist]
        md += f"Min distance: {np.min(dists_only)}, Max distance: {np.max(dists_only)}\n"
        
    with open('tasks/astar-island/gemini_replay_analysis.md', 'w') as fh:
        fh.write(md)

    print("Wrote analysis to tasks/astar-island/gemini_replay_analysis.md")

if __name__ == '__main__':
    analyze_replays()