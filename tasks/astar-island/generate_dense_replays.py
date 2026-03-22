import json
import requests
import time
import os
from pathlib import Path

def fetch_dense_training_data():
    with open("tasks/astar-island/.token") as f:
        token = f.read().strip()

    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    s.cookies.set("access_token", token)

    out_dir = Path("tasks/astar-island/replays/dense_training")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Fetch rounds
    res = s.get("https://api.ainm.no/astar-island/rounds")
    if res.status_code != 200:
        print("Failed to get rounds:", res.text)
        return
        
    rounds = res.json()
    
    # Target the last 3 completed rounds to get a mix of regimes
    completed_rounds = [r for r in rounds if r["status"] == "completed"]
    # Sort by round number descending and pick top 3
    target_rounds = sorted(completed_rounds, key=lambda x: x["round_number"], reverse=True)[:3]
    
    # We will fetch 10 deterministic trajectories per seed (sim_seeds 1 through 10)
    SAMPLES_PER_SEED = 10
    
    print(f"Generating dense training data for Rounds: {[r['round_number'] for r in target_rounds]}")
    print(f"Targeting {SAMPLES_PER_SEED} fixed sim_seeds per map (Total {len(target_rounds) * 5 * SAMPLES_PER_SEED} replays)")
    
    for rnd in target_rounds:
        r_num = rnd["round_number"]
        r_id = rnd["id"]
        
        for seed_idx in range(5):
            for sim_seed in range(1, SAMPLES_PER_SEED + 1):
                out_file = out_dir / f"round{r_num}_seed{seed_idx}_sim{sim_seed}.json"
                if out_file.exists():
                    continue
                    
                print(f"Fetching R{r_num} Seed {seed_idx} [SimSeed {sim_seed}]...")
                
                while True:
                    resp = s.post("https://api.ainm.no/astar-island/replay", json={
                        "round_id": r_id,
                        "seed_index": seed_idx,
                        "sim_seed": sim_seed
                    })
                    
                    if resp.status_code == 200:
                        with open(out_file, "w") as f:
                            json.dump(resp.json(), f)
                        # Polite delay as requested
                        time.sleep(0.5)
                        break
                    elif resp.status_code == 429:
                        print("  -> Rate limited! Sleeping 5 seconds...")
                        time.sleep(5.0)
                    else:
                        print(f"  -> Failed: {resp.status_code}")
                        time.sleep(1.0)
                        break

if __name__ == "__main__":
    fetch_dense_training_data()
