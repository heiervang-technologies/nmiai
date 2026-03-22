import json
import glob
import numpy as np
from collections import defaultdict
from scipy.ndimage import convolve

def extract_remaining_rules():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f]
    
    print(f"Extracting remaining CA rules from {len(files)} replays...")

    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)
    
    # 0: Empty, 1: Settle, 2: Port, 3: Ruin, 4: Forest, 5: Mount, 10: Ocean, 11: Plains
    
    # Trackers for Port and Forest
    port_sources = defaultdict(int)
    port_formation_from_settle = {'ocean_neighbors': defaultdict(lambda: [0, 0])} # count -> [became_port, total_stayed_or_became_port]
    forest_clearing = {'civ_neighbors': defaultdict(lambda: [0, 0])} # count -> [became_settle, total_stayed_or_became_settle]
    
    # Track phase-dependent growth
    phase_growth = defaultdict(lambda: defaultdict(lambda: [0,0])) # phase -> civ_neighbors -> [became_settle, total]

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 2: continue
        
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            mod4 = i % 4
            
            # Neighbors
            is_civ = ((prev == 1) | (prev == 2)).astype(np.int32)
            n_civ = convolve(is_civ, kernel, mode='constant')
            
            is_ocean = (prev == 10).astype(np.int32)
            n_ocean = convolve(is_ocean, kernel, mode='constant')
            
            # Analyze Ports
            new_ports = (curr == 2) & (prev != 2)
            if np.any(new_ports):
                y_idx, x_idx = np.where(new_ports)
                for y, x in zip(y_idx, x_idx):
                    port_sources[int(prev[y, x])] += 1
            
            # Port formation from settlements specifically
            settles = (prev == 1)
            if np.any(settles):
                y_idx, x_idx = np.where(settles)
                for y, x in zip(y_idx, x_idx):
                    oc_c = n_ocean[y, x]
                    # We only care about cases where it didn't collapse into a ruin
                    if curr[y, x] in (1, 2):
                        port_formation_from_settle['ocean_neighbors'][oc_c][1] += 1
                        if curr[y, x] == 2:
                            port_formation_from_settle['ocean_neighbors'][oc_c][0] += 1
                            
            # Forest clearing
            forests = (prev == 4)
            if np.any(forests):
                y_idx, x_idx = np.where(forests)
                # Sample 5% of forests for performance if there are too many
                if len(y_idx) > 1000:
                    indices = np.random.choice(len(y_idx), 1000, replace=False)
                else:
                    indices = range(len(y_idx))
                    
                for idx in indices:
                    y = y_idx[idx]
                    x = x_idx[idx]
                    cv_c = n_civ[y, x]
                    forest_clearing['civ_neighbors'][cv_c][1] += 1
                    if curr[y, x] == 1:
                        forest_clearing['civ_neighbors'][cv_c][0] += 1
                        
            # Phase-dependent growth (Plains -> Settle)
            plains = (prev == 11)
            if np.any(plains):
                y_idx, x_idx = np.where(plains)
                if len(y_idx) > 1000:
                    indices = np.random.choice(len(y_idx), 1000, replace=False)
                else:
                    indices = range(len(y_idx))
                
                for idx in indices:
                    y = y_idx[idx]
                    x = x_idx[idx]
                    cv_c = n_civ[y, x]
                    phase_growth[mod4][cv_c][1] += 1
                    if curr[y, x] == 1:
                        phase_growth[mod4][cv_c][0] += 1

    md = "# Remaining CA Transition Rules\n\n"
    
    md += "## 1. Port Formation\n"
    md += "Where do Ports come from?\n"
    class_map = {0: 'Empty', 1: 'Settle', 2: 'Port', 3: 'Ruin', 4: 'Forest', 5: 'Mount', 10: 'Ocean', 11: 'Plains'}
    total_ports = sum(port_sources.values())
    for k, v in port_sources.items():
        md += f"- From {class_map.get(k, k)}: {v} ({v/total_ports*100:.1f}%)\n"
        
    md += "\nFor existing Settlements, what is the probability of upgrading to a Port based on Ocean neighbors?\n"
    md += "| Ocean Neighbors | P(Upgrade to Port) | Sample Size |\n"
    md += "|---|---|---|\n"
    for oc in sorted(port_formation_from_settle['ocean_neighbors'].keys()):
        became, tot = port_formation_from_settle['ocean_neighbors'][oc]
        if tot > 50:
            md += f"| {oc} | {became/tot:.4f} | {tot} |\n"

    md += "\n## 2. Forest Clearing\n"
    md += "Probability of a Forest turning into a Settlement by Civ Neighbors:\n"
    md += "| Civ Neighbors | P(Clear Forest) | Sample Size |\n"
    md += "|---|---|---|\n"
    for cv in sorted(forest_clearing['civ_neighbors'].keys()):
        became, tot = forest_clearing['civ_neighbors'][cv]
        if tot > 50:
            md += f"| {cv} | {became/tot:.4f} | {tot} |\n"

    md += "\n## 3. Phase-Dependent Growth Rates\n"
    md += "Does the global 4-step cycle change the actual probabilities of growth, or just the number of eligible cells?\n"
    md += "Looking at Plains -> Settlement for 1, 2, and 3 civ neighbors across the 4 phases:\n"
    md += "| Civ Neighbors | Phase 0 (Growth) | Phase 1 | Phase 2 | Phase 3 |\n"
    md += "|---|---|---|---|---|\n"
    for cv in [1, 2, 3]:
        rates = []
        for p in [0, 1, 2, 3]:
            became, tot = phase_growth[p][cv]
            rate = became/tot if tot > 0 else 0
            rates.append(f"{rate:.4f} (n={tot})")
        md += f"| {cv} | {rates[0]} | {rates[1]} | {rates[2]} | {rates[3]} |\n"

    with open('tasks/astar-island/remaining_rules.md', 'w') as f:
        f.write(md)
        
    print("Rule extraction complete! Wrote to tasks/astar-island/remaining_rules.md")

if __name__ == '__main__':
    extract_remaining_rules()