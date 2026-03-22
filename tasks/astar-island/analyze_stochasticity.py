import pickle
import numpy as np

with open('tasks/astar-island/extracted_tensor_stats.pkl', 'rb') as f:
    transitions = pickle.load(f)

print("Analyzing most common stochastic transitions...")

class_map = {0: 'Empty', 1: 'Settle', 2: 'Port', 3: 'Ruin', 4: 'Forest', 5: 'Mount', 10: 'Ocean', 11: 'Plains'}

# Filter stochastic transitions with at least 100 samples
stoch_trans = []
for k, counts in transitions.items():
    s = counts.sum()
    if s >= 100 and np.max(counts) < s:
        stoch_trans.append((k, counts, s))

# Sort by sample size descending
stoch_trans.sort(key=lambda x: x[2], reverse=True)

for i in range(min(20, len(stoch_trans))):
    k, counts, s = stoch_trans[i]
    phase, regime, center, c3, c7, o3, f3, r3 = k
    
    reg_str = ["Harsh", "Moderate", "Prosperous"][regime]
    print(f"\nState: {class_map.get(center, center)} (Phase {phase}, {reg_str}, civ3:{c3}, civ7:{c7}, oc3:{o3}, for3:{f3}, ruin3:{r3}) - {s} samples")
    
    probs = counts / s
    for nxt in range(12):
        if counts[nxt] > 0:
            print(f"  -> {class_map.get(nxt, nxt)}: {probs[nxt]:.4f} ({counts[nxt]} samples)")

