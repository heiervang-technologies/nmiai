import json
import glob
import numpy as np
from scipy.ndimage import convolve
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq

def extract_dataset():
    files = glob.glob('tasks/astar-island/replays/*.json') + glob.glob('tasks/astar-island/replays_expanded/*.json')
    print(f"Extracting flat transition dataset from {len(files)} replays...")

    k3 = np.ones((3, 3), dtype=np.int32)
    k3[1, 1] = 0
    k7 = np.ones((7, 7), dtype=np.int32)
    k7[3, 3] = 0
    
    # Store batches to avoid OOM
    batch_size = 50
    all_data = []
    
    output_file = 'tasks/astar-island/transitions_dataset.parquet'
    if os.path.exists(output_file):
        os.remove(output_file)

    parquet_schema = None
    writer = None

    for f_idx, f in enumerate(files):
        rnd = int(os.path.basename(f).split('_')[0].replace('round', ''))
        
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 51: continue
        
        for i in range(1, len(frames)):
            prev = np.array(frames[i-1]['grid'])
            curr = np.array(frames[i]['grid'])
            phase = i % 4
            
            is_civ = ((prev == 1) | (prev == 2)).astype(np.int32)
            c3 = convolve(is_civ, k3, mode='constant').flatten()
            c7 = convolve(is_civ, k7, mode='constant')
            c7 = np.clip(c7, 0, 25).flatten()
            
            is_ocean = (prev == 10).astype(np.int32)
            o3 = convolve(is_ocean, k3, mode='constant').flatten()
            
            is_forest = (prev == 4).astype(np.int32)
            f3 = convolve(is_forest, k3, mode='constant').flatten()
            
            state = prev.flatten()
            next_state = curr.flatten()
            
            # Sampling trivial static empty/ocean cells (state 0 or 10 or 11 with no neighbors)
            # They dominate the dataset but don't teach the model anything interesting.
            is_trivial = (state == next_state) & ((state == 0) | (state == 10) | (state == 11)) & (c3 == 0) & (c7 == 0) & (o3 == 0) & (f3 == 0)
            
            # Keep 1% of trivial, all non-trivial
            keep_mask = ~is_trivial | (np.random.rand(len(state)) < 0.01)
            
            weights = np.ones(len(state), dtype=np.float32)
            weights[is_trivial & keep_mask] = 100.0 # Weight the sampled trivial cells back up
            
            batch_data = {
                'round': np.full(keep_mask.sum(), rnd, dtype=np.int16),
                'phase': np.full(keep_mask.sum(), phase, dtype=np.int8),
                'state': state[keep_mask].astype(np.int8),
                'c3': c3[keep_mask].astype(np.int8),
                'c7': c7[keep_mask].astype(np.int8),
                'o3': o3[keep_mask].astype(np.int8),
                'f3': f3[keep_mask].astype(np.int8),
                'next_state': next_state[keep_mask].astype(np.int8),
                'weight': weights[keep_mask]
            }
            all_data.append(pd.DataFrame(batch_data))
            
        if (f_idx + 1) % batch_size == 0 or (f_idx + 1) == len(files):
            df_batch = pd.concat(all_data, ignore_index=True)
            table = pa.Table.from_pandas(df_batch)
            
            if writer is None:
                parquet_schema = table.schema
                writer = pq.ParquetWriter(output_file, parquet_schema)
            
            writer.write_table(table)
            all_data = []
            print(f"Processed {f_idx + 1} / {len(files)} files. Written to parquet.")

    if writer is not None:
        writer.close()
    
    print(f"Extraction complete! Dataset saved to {output_file}")

if __name__ == '__main__':
    extract_dataset()
