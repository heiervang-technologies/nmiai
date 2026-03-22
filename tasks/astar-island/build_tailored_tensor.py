import json
import torch
import numpy as np
import sys
import os
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
import gpu_simulator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_regime(final_grid):
    civ_count = np.sum((final_grid == 1) | (final_grid == 2))
    if civ_count < 30: return 0
    if civ_count > 150: return 2
    return 1

def build_tailored_tensor(ig, lookup_tensor, num_sims=1000):
    H, W = ig.shape
    has_ruin_dim = lookup_tensor.dim() >= 8

    # Best-matched regime based on ig
    regime = get_regime(ig)
    print(f"  Using matched regime {regime}")
    if lookup_tensor.dim() == 8 and lookup_tensor.shape[0] == 3:
        # Use exact regime instead of blending
        phase_lookups = lookup_tensor[regime].to(device)
    else:
        phase_lookups = lookup_tensor.to(device)

    # Run simulations and collect transitions
    state = torch.tensor(ig, device=device, dtype=torch.long).unsqueeze(0).repeat(num_sims, 1, 1)

    k3 = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
    k3[0, 0, 1, 1] = 0
    k7 = torch.ones((1, 1, 7, 7), device=device, dtype=torch.float32)
    k7[0, 0, 3, 3] = 0

    # Dimensions: P=4, ST=12, C3=9, C7=26, O3=9, F3=9, NXT=12
    counts_full = torch.zeros((4, 12, 9, 26, 9, 9, 12), device=device, dtype=torch.float32)

    for step in range(1, 51):
        phase = step % 4

        is_civ = ((state == 1) | (state == 2)).float().unsqueeze(1)
        n_civ3 = F.conv2d(is_civ, k3, padding=1).squeeze(1).long()
        n_civ3 = torch.clamp(n_civ3, 0, 8)

        n_civ7 = F.conv2d(is_civ, k7, padding=3).squeeze(1).long()
        n_civ7 = torch.clamp(n_civ7, 0, 25)

        is_ocean = (state == 10).float().unsqueeze(1)
        n_ocean = F.conv2d(is_ocean, k3, padding=1).squeeze(1).long()
        n_ocean = torch.clamp(n_ocean, 0, 8)

        is_forest = (state == 4).float().unsqueeze(1)
        n_forest = F.conv2d(is_forest, k3, padding=1).squeeze(1).long()
        n_forest = torch.clamp(n_forest, 0, 8)

        phase_lookup = phase_lookups[phase]
        flat_state = torch.clamp(state.view(-1), 0, 11)
        flat_c3 = torch.clamp(n_civ3.view(-1), 0, 8)
        flat_c7 = torch.clamp(n_civ7.view(-1), 0, 25)
        flat_o3 = torch.clamp(n_ocean.view(-1), 0, 8)
        flat_f3 = torch.clamp(n_forest.view(-1), 0, 8)

        probs = phase_lookup[flat_state, flat_c3, flat_c7, flat_o3, flat_f3]
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        sampled = torch.multinomial(probs, 1).squeeze(-1)
        next_state = sampled.view(num_sims, H, W)
        
        # Accumulate transitions into counts_full
        flat_next = next_state.view(-1)
        
        idx = flat_next + 12 * (flat_f3 + 9 * (flat_o3 + 9 * (flat_c7 + 26 * (flat_c3 + 9 * flat_state))))
        
        counts = torch.bincount(idx, minlength=12*9*26*9*9*12)
        counts = counts.view(12, 9, 26, 9, 9, 12).float()
        
        counts_full[phase] += counts
        state = next_state

    # Normalize to probabilities
    sums = counts_full.sum(dim=-1, keepdim=True)
    tailored_tensor = torch.where(sums > 0, counts_full / sums, phase_lookups)
    
    return tailored_tensor

def main():
    import requests
    with open('tasks/astar-island/.token') as f:
        token = f.read().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    
    rounds = s.get("https://api.ainm.no/astar-island/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round.")
        sys.exit(0)
        
    details = s.get(f"https://api.ainm.no/astar-island/rounds/{active['id']}").json()
    initial_states = details["initial_states"]
    
    print("Loading dense_lookup.pt...")
    lookup_tensor = torch.load('tasks/astar-island/dense_lookup.pt', map_location='cpu')
    
    out_dir = 'tasks/astar-island/round_tensors'
    os.makedirs(out_dir, exist_ok=True)
    
    for seed_idx, state_data in enumerate(initial_states):
        ig = np.array(state_data["grid"])
        print(f"Building tailored tensor for seed {seed_idx}...")
        tt = build_tailored_tensor(ig, lookup_tensor, num_sims=1000)
        
        out_path = os.path.join(out_dir, f'tailored_tensor_seed{seed_idx}.pt')
        torch.save(tt.cpu(), out_path)
        print(f"  Saved {out_path}")

if __name__ == '__main__':
    main()
