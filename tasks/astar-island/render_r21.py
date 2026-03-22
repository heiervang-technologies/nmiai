import json
import os
import sys
import numpy as np
import requests

def get_active_round_details():
    try:
        with open("tasks/astar-island/.token") as f:
            token = f.read().strip()
    except Exception:
        print("Could not read token")
        return None

    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    s.cookies.set("access_token", token)

    rounds = s.get("https://api.ainm.no/astar-island/rounds").json()
    active_round = next((r for r in rounds if r["status"] == "active"), None)
    if not active_round:
        print("No active round found")
        return None

    return s.get(f"https://api.ainm.no/astar-island/rounds/{active_round['id']}").json()

def print_qualitative_analysis(details):
    if not details: return
    
    round_num = details["round_number"]
    print(f"=== QUALITATIVE ANALYSIS FOR R{round_num} ===")
    
    class_map = {0: '.', 1: 'S', 2: 'P', 3: 'R', 4: 'F', 5: 'M', 10: 'O', 11: '-'}
    
    for seed_idx, initial_state in enumerate(details["initial_states"]):
        if seed_idx > 0: break # Just do seed 0 for now to avoid massive output
        
        ig = np.array(initial_state["grid"], dtype=np.int32)
        
        obs_file = f"tasks/astar-island/logs/round{round_num}/observations_seed{seed_idx}.json"
        if os.path.exists(obs_file):
            with open(obs_file) as f:
                obs_list = json.load(f)
        else:
            obs_list = []
            
        print(f"\nSeed {seed_idx}: Found {len(obs_list)} viewports")
        
        # We will render the initial grid, but overlay the observed GT where available.
        # We'll use ANSI colors to highlight observed cells.
        # Initial grid is dim, observed cells are bright
        
        observed_mask = np.zeros((40, 40), dtype=bool)
        obs_grid = np.zeros((40, 40), dtype=np.int32)
        
        for obs in obs_list:
            vx, vy = obs["viewport_x"], obs["viewport_y"]
            g = np.array(obs["grid"])
            oh, ow = g.shape
            for y in range(oh):
                for x in range(ow):
                    if vy + y < 40 and vx + x < 40:
                        observed_mask[vy + y, vx + x] = True
                        obs_grid[vy + y, vx + x] = g[y, x]
        
        print("\nMap Overlay (Dim = Initial State, Bright/Brackets = Observed Final State):")
        for y in range(40):
            row_str = ""
            for x in range(40):
                if observed_mask[y, x]:
                    char = class_map.get(obs_grid[y, x], '?')
                    row_str += f"[{char}]"
                else:
                    char = class_map.get(ig[y, x], '?')
                    row_str += f" {char} "
            print(row_str)
            
        # Quick stats on observed changes
        changes = 0
        observed_cells = 0
        for y in range(40):
            for x in range(40):
                if observed_mask[y, x]:
                    observed_cells += 1
                    # Note: obs_grid contains the raw class code or 6-class format? 
                    # Usually viewports are raw codes (0,1,2,3,4,5,10,11)
                    if obs_grid[y, x] != ig[y, x]:
                        changes += 1
                        
        print(f"\nObserved Cells: {observed_cells} / 1600 ({observed_cells/1600*100:.1f}%)")
        if observed_cells > 0:
            print(f"Observed Changes vs Initial: {changes} ({changes/observed_cells*100:.1f}%)")
            
        # Estimate global regime from observed sample
        # If we saw X settlements in the viewports, extrapolate to full map
        obs_settle = np.sum((obs_grid == 1) | (obs_grid == 2))
        est_total = int(obs_settle * (1600 / max(1, observed_cells)))
        print(f"Extrapolated Total Settlements: ~{est_total} -> Likely Regime: {'Harsh' if est_total < 30 else 'Prosperous' if est_total > 150 else 'Moderate'}")

if __name__ == "__main__":
    details = get_active_round_details()
    print_qualitative_analysis(details)
