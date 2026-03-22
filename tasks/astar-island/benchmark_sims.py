import json
import numpy as np
import torch
import time
import gpu_exact_mc_v2

def run_benchmark():
    with open('tasks/astar-island/ground_truth/round16_seed0.json') as f:
        d = json.load(f)
    ig = np.array(d['initial_grid'], dtype=np.int32)
    db = torch.load('tasks/astar-island/gpu_search_tensor.pt', weights_only=False)
    
    print("Benchmarking num_sims to max out GPU within timeout...")
    for sims in [5000, 10000, 15000, 20000, 30000, 40000]:
        t0 = time.time()
        # We only need to run the pure MC function
        try:
            pred = gpu_exact_mc_v2.run_gpu_mc(ig, db, regime=1, num_sims=sims)
            t1 = time.time()
            print(f"num_sims: {sims} took {t1-t0:.2f} seconds")
        except RuntimeError as e: # Catch OOM
            print(f"num_sims: {sims} caused OOM: {e}")
            break

if __name__ == '__main__':
    run_benchmark()
