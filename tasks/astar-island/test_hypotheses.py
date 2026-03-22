import json
import glob
import numpy as np
from collections import defaultdict
from scipy.ndimage import distance_transform_edt, convolve
import os

def test_hypotheses():
    files = [f for f in glob.glob('tasks/astar-island/replays/round*_seed*.json') if 'dense_training' not in f and 'simseed' not in f]
    print(f"Testing on {len(files)} base replay files.")

    # Data structures for tracking
    # H1
    h1_transitions = {0: defaultdict(int), 1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int)}
    
    # H2
    h2_early_changes = []
    h2_late_changes = []
    
    # H3
    h3_prosperous_growth = []
    h3_moderate_growth = []
    h3_harsh_growth = []
    
    def get_regime(final_grid):
        civ_count = np.sum((final_grid == 1) | (final_grid == 2))
        if civ_count < 30: return 'Harsh'
        if civ_count > 150: return 'Prosperous'
        return 'Moderate'

    # H4
    h4_ruin_spawns = 0
    h4_ruin_persists = 0
    
    # H5
    h5_new_settlements_mod3 = 0
    h5_adj_settlements_mod3 = 0
    h5_new_settlements_other = 0
    h5_adj_settlements_other = 0
    
    # H6
    h6_early_growth_by_regime = defaultdict(list)
    
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
    class_map = {0: 'Empty', 1: 'Settle', 2: 'Port', 3: 'Ruin', 4: 'Forest', 5: 'Mount', 10: 'Ocean', 11: 'Plains'}

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 51: continue
        
        final_grid = np.array(frames[-1]['grid'])
        regime = get_regime(final_grid)
        
        early_growth = 0
        
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            mod4 = i % 4
            
            diff = prev != curr
            num_changes = np.sum(diff)
            
            # H1
            if num_changes > 0:
                y_idx, x_idx = np.where(diff)
                for y, x in zip(y_idx, x_idx):
                    old_v = prev[y, x]
                    new_v = curr[y, x]
                    t_str = f"{class_map.get(old_v, old_v)}->{class_map.get(new_v, new_v)}"
                    h1_transitions[mod4][t_str] += 1
                    
            # H2
            if i <= 10: h2_early_changes.append(num_changes)
            if i >= 40: h2_late_changes.append(num_changes)
            
            # H3
            new_settle = (curr == 1) & (prev != 1)
            num_new_settle = np.sum(new_settle)
            if mod4 == 3:
                if regime == 'Prosperous': h3_prosperous_growth.append(num_new_settle)
                elif regime == 'Moderate': h3_moderate_growth.append(num_new_settle)
                else: h3_harsh_growth.append(num_new_settle)
                
            # H4
            if i < len(frames) - 1:
                next_f = np.array(frames[i+1]['grid'])
                new_ruins = (curr == 3) & (prev != 3)
                if np.any(new_ruins):
                    h4_ruin_spawns += np.sum(new_ruins)
                    h4_ruin_persists += np.sum(next_f[new_ruins] == 3)
                    
            # H5
            if num_new_settle > 0:
                civ_prev = (prev == 1) | (prev == 2)
                civ_neighbors = convolve(civ_prev.astype(int), kernel, mode='constant')
                adj_count = np.sum((civ_neighbors > 0) & new_settle)
                if mod4 == 3:
                    h5_new_settlements_mod3 += num_new_settle
                    h5_adj_settlements_mod3 += adj_count
                else:
                    h5_new_settlements_other += num_new_settle
                    h5_adj_settlements_other += adj_count
                    
            # H6 (steps 1-4, note 0 is initial)
            if i <= 4:
                early_growth += num_new_settle
                
        h6_early_growth_by_regime[regime].append(early_growth)
        
    # Formatting output
    md = "# Hypothesis Testing Results (Replay Data)\n\n"
    
    # H1
    md += "## HYPOTHESIS 1: The 4-phase cycle\n"
    for m in [0, 1, 2, 3]:
        md += f"**Mod 4 == {m}**\n"
        top = sorted(h1_transitions[m].items(), key=lambda x: x[1], reverse=True)[:3]
        for t, c in top: md += f"- {t}: {c}\n"
        md += "\n"
        
    # H2
    e_mean = np.mean(h2_early_changes)
    l_mean = np.mean(h2_late_changes)
    md += f"## HYPOTHESIS 2: Late-game acceleration\n"
    md += f"- Mean changes Steps 1-10: {e_mean:.2f}\n"
    md += f"- Mean changes Steps 40-49: {l_mean:.2f}\n"
    md += f"- Ratio: {l_mean/e_mean:.2f}x\n\n"
    
    # H3
    p_g = np.mean(h3_prosperous_growth)
    m_g = np.mean(h3_moderate_growth)
    h_g = np.mean(h3_harsh_growth)
    md += f"## HYPOTHESIS 3: Prosperous vs Moderate growth rate\n"
    md += f"- Prosperous Mod 4=3 Mean Growth: {p_g:.2f}\n"
    md += f"- Moderate Mod 4=3 Mean Growth: {m_g:.2f}\n"
    md += f"- Harsh Mod 4=3 Mean Growth: {h_g:.2f}\n"
    if m_g > 0:
        md += f"- Ratio (Prosperous / Moderate): {p_g/m_g:.2f}x\n\n"
    else: md += "\n"
    
    # H4
    md += f"## HYPOTHESIS 4: Ruins Persistence\n"
    md += f"- Total Ruin Spawns: {h4_ruin_spawns}\n"
    md += f"- Ruins Persisting 2+ steps: {h4_ruin_persists}\n\n"
    
    # H5
    mod3_pct = (h5_adj_settlements_mod3 / h5_new_settlements_mod3) * 100 if h5_new_settlements_mod3 > 0 else 0
    other_pct = (h5_adj_settlements_other / h5_new_settlements_other) * 100 if h5_new_settlements_other > 0 else 0
    md += f"## HYPOTHESIS 5: Settlement Expansion Locality\n"
    md += f"- Mod 4=3: {h5_adj_settlements_mod3}/{h5_new_settlements_mod3} ({mod3_pct:.1f}%) were adjacent to existing settlements.\n"
    md += f"- Other steps: {h5_adj_settlements_other}/{h5_new_settlements_other} ({other_pct:.1f}%) were adjacent.\n\n"
    
    # H6
    md += f"## HYPOTHESIS 6: Regime prediction from steps 1-4\n"
    md += "Average new settlements spawned in the first 4 steps by final regime:\n"
    for r in ['Prosperous', 'Moderate', 'Harsh']:
        if h6_early_growth_by_regime[r]:
            md += f"- {r}: {np.mean(h6_early_growth_by_regime[r]):.2f} (min: {np.min(h6_early_growth_by_regime[r])}, max: {np.max(h6_early_growth_by_regime[r])})\n"

    with open('tasks/astar-island/hypothesis_test_results.md', 'w') as f:
        f.write(md)
        
    print("Testing complete. Results saved to hypothesis_test_results.md")

if __name__ == '__main__':
    test_hypotheses()
