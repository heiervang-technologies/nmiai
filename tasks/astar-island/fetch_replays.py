import json
import requests
import os
import glob
import time
from pathlib import Path

def fetch_replays():
    with open("tasks/astar-island/.token") as f:
        token = f.read().strip()

    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    s.cookies.set("access_token", token)

    out_dir = Path("tasks/astar-island/replays")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Get list of rounds to get their UUIDs
    # We can fetch round details from existing ground_truth or rounds api
    res = s.get("https://api.ainm.no/astar-island/rounds")
    if res.status_code != 200:
        print("Failed to get rounds:", res.text)
        return
        
    rounds = res.json()
    print(f"Found {len(rounds)} rounds.")
    
    for rnd in rounds:
        r_num = rnd["round_number"]
        r_id = rnd["id"]
        # Skip rounds that are not completed maybe? Or just try to get all
        if rnd["status"] != "completed":
            print(f"Skipping R{r_num} (status: {rnd['status']})")
            continue
            
        for seed_idx in range(5):
            out_file = out_dir / f"round{r_num}_seed{seed_idx}.json"
            if out_file.exists():
                continue
                
            print(f"Fetching replay for R{r_num} Seed {seed_idx}...")
            while True:
                resp = s.post("https://api.ainm.no/astar-island/replay", json={
                    "round_id": r_id,
                    "seed_index": seed_idx
                })
                if resp.status_code == 200:
                    with open(out_file, "w") as f:
                        json.dump(resp.json(), f)
                    break
                elif resp.status_code == 429:
                    print("  -> 429 Rate limited, sleeping 1s...")
                    time.sleep(1)
                else:
                    print(f"  -> Failed: {resp.status_code}")
                    break
            time.sleep(0.3)

if __name__ == "__main__":
    fetch_replays()
