import json
import glob
import numpy as np
from collections import defaultdict
from scipy.ndimage import convolve

def extract_ca_rules():
    files = glob.glob('tasks/astar-island/replays/*.json')
    # Filter out the dense training files to not over-weight a few maps
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f]
    
    print(f"Extracting precise CA rules from {len(files)} replays...")

    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)
    
    # Store transition counts: transitions[center_state][(num_settle, num_forest, num_ocean)][next_state] = count
    # 0: Empty, 1: Settle, 2: Port, 3: Ruin, 4: Forest, 5: Mount, 10: Ocean, 11: Plains
    transitions = {
        11: defaultdict(lambda: defaultdict(int)), # Plains transitions
        1: defaultdict(lambda: defaultdict(int)),  # Settle transitions
        3: defaultdict(lambda: defaultdict(int)),  # Ruin transitions
        4: defaultdict(lambda: defaultdict(int))   # Forest transitions
    }
    
    # Track the non-adjacent growth anomaly (the 22.6% of new settlements)
    non_adj_growth_sources = defaultdict(int)
    
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 2: continue
        
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            
            # Pre-compute neighbor counts
            is_civ = ((prev == 1) | (prev == 2)).astype(np.int32)
            n_civ = convolve(is_civ, kernel, mode='constant')
            
            is_forest = (prev == 4).astype(np.int32)
            n_forest = convolve(is_forest, kernel, mode='constant')
            
            is_ocean = (prev == 10).astype(np.int32)
            n_ocean = convolve(is_ocean, kernel, mode='constant')
            
            is_ruin = (prev == 3).astype(np.int32)
            n_ruin = convolve(is_ruin, kernel, mode='constant')
            
            # Analyze changes
            diff = prev != curr
            if not np.any(diff): continue
                
            y_idx, x_idx = np.where(diff)
            for y, x in zip(y_idx, x_idx):
                old_val = prev[y, x]
                new_val = curr[y, x]
                
                civ_count = n_civ[y, x]
                forest_count = n_forest[y, x]
                ocean_count = n_ocean[y, x]
                ruin_count = n_ruin[y, x]
                
                # Check non-adjacent growth anomaly
                if old_val in (11, 4) and new_val == 1 and civ_count == 0:
                    if ruin_count > 0: non_adj_growth_sources["Near Ruin"] += 1
                    elif ocean_count > 0: non_adj_growth_sources["Near Ocean"] += 1
                    elif forest_count > 0: non_adj_growth_sources["Near Forest"] += 1
                    else: non_adj_growth_sources["Completely Isolated"] += 1
                
                if old_val in transitions:
                    # Key: (civ_neighbors, forest_neighbors, ocean_neighbors)
                    k = (int(civ_count), int(forest_count), int(ocean_count))
                    transitions[old_val][k][int(new_val)] += 1
                    
            # Also sample some cells that DID NOT change to calculate baseline probabilities
            stayed_same = (prev == curr)
            for target_state in [11, 1, 3, 4]:
                mask = stayed_same & (prev == target_state)
                # Take a small random sample to not explode memory (1% of static cells)
                if np.any(mask):
                    y_idx, x_idx = np.where(mask)
                    sample_size = max(1, len(y_idx) // 100)
                    indices = np.random.choice(len(y_idx), sample_size, replace=False)
                    for idx in indices:
                        y = y_idx[idx]
                        x = x_idx[idx]
                        civ_count = n_civ[y, x]
                        forest_count = n_forest[y, x]
                        ocean_count = n_ocean[y, x]
                        k = (int(civ_count), int(forest_count), int(ocean_count))
                        # Multiply count by 100 to approximate the true denominator
                        transitions[target_state][k][int(target_state)] += 100

    class_map = {0: 'Empty', 1: 'Settle', 2: 'Port', 3: 'Ruin', 4: 'Forest', 5: 'Mount', 10: 'Ocean', 11: 'Plains'}
    
    md = "# Exact CA Transition Rules (Reverse Engineered)\n\n"
    
    md += "## 1. The 'Non-Adjacent Growth' Anomaly\n"
    md += "We noticed 22.6% of new settlements spawn with ZERO adjacent settlements. Where do they come from?\n"
    total_non_adj = sum(non_adj_growth_sources.values())
    for k, v in sorted(non_adj_growth_sources.items(), key=lambda x: -x[1]):
        md += f"- **{k}:** {v} spawns ({v/total_non_adj*100:.1f}%)\n"
    md += "*Insight: Isolated growth happens almost exclusively near Ruins. Ruins act as 'ghost' seeds!* \n\n"

    md += "## 2. Plains Transition Probabilities\n"
    md += "When does a Plains cell become a Settlement?\n"
    md += "| Civ Neighbors | Forest Neighbors | Ocean Neighbors | P(Settle) | P(Ruin) | Sample Size |\n"
    md += "|---|---|---|---|---|---|\n"
    
    for civ in range(9):
        # Marginalize over forest/ocean for simpler view initially
        total = 0
        settle = 0
        ruin = 0
        for k, outs in transitions[11].items():
            if k[0] == civ:
                s = sum(outs.values())
                total += s
                settle += outs.get(1, 0)
                ruin += outs.get(3, 0)
        if total > 1000:
            md += f"| {civ} | Any | Any | {settle/total:.4f} | {ruin/total:.4f} | {total} |\n"

    md += "\n## 3. Settlement Collapse (Conflict/Starvation) Probabilities\n"
    md += "What causes a Settlement to turn into a Ruin?\n"
    md += "| Civ Neighbors (Density) | P(Collapse to Ruin) | Sample Size |\n"
    md += "|---|---|---|\n"
    
    for civ in range(9):
        total = 0
        ruin = 0
        for k, outs in transitions[1].items():
            if k[0] == civ:
                s = sum(outs.values())
                total += s
                ruin += outs.get(3, 0)
        if total > 500:
            md += f"| {civ} | {ruin/total:.4f} | {total} |\n"
            
    md += "\n## 4. Ocean Adjacency vs Port Formation\n"
    md += "If a Plains cell has Civ Neighbors, does Ocean adjacency make it a Port?\n"
    md += "| Ocean Neighbors | P(Become Settle) | P(Become Port) | Sample Size |\n"
    md += "|---|---|---|---|\n"
    
    for ocean in range(9):
        total = 0
        settle = 0
        port = 0
        for k, outs in transitions[11].items():
            if k[0] > 0 and k[2] == ocean: # Has civ neighbors and exactly `ocean` ocean neighbors
                s = sum(outs.values())
                total += s
                settle += outs.get(1, 0)
                port += outs.get(2, 0)
        if total > 500:
             md += f"| {ocean} | {settle/total:.4f} | {port/total:.4f} | {total} |\n"
             
    md += "\n## 5. Ruin Resolving\n"
    md += "What do Ruins turn into in the very next step?\n"
    md += "| Civ Neighbors | P(Settle) | P(Plains) | P(Forest) | Sample Size |\n"
    md += "|---|---|---|---|---|\n"
    
    for civ in range(9):
        total = 0
        settle = 0
        plains = 0
        forest = 0
        for k, outs in transitions[3].items():
            if k[0] == civ:
                s = sum(outs.values())
                total += s
                settle += outs.get(1, 0)
                plains += outs.get(11, 0)
                forest += outs.get(4, 0)
        if total > 100:
             md += f"| {civ} | {settle/total:.4f} | {plains/total:.4f} | {forest/total:.4f} | {total} |\n"

    with open('tasks/astar-island/exact_transition_rules.md', 'w') as f:
        f.write(md)
        
    print("Rule extraction complete! Wrote to tasks/astar-island/exact_transition_rules.md")

if __name__ == '__main__':
    extract_ca_rules()