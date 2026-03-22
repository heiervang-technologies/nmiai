import torch
import numpy as np

def compile():
    print("Loading sparse dictionary...")
    transitions = torch.load('tasks/astar-island/sparse_lookup_v2.pt', weights_only=False)
    
    keys = []
    probs = []
    
    # Dimensions:
    # phase: 4
    # regime: 3
    # center: 12
    # c3: 9
    # c7: 26
    # o3: 9
    # f3: 9
    # r3: 9
    # p3: 9
    
    # We will compute a flat integer key.
    # To fit in torch.int64, we use multipliers:
    m_phase = 1
    m_regime = 4
    m_center = 4 * 3
    m_c3 = 4 * 3 * 12
    m_c7 = 4 * 3 * 12 * 9
    m_o3 = 4 * 3 * 12 * 9 * 26
    m_f3 = 4 * 3 * 12 * 9 * 26 * 9
    m_r3 = 4 * 3 * 12 * 9 * 26 * 9 * 9
    m_p3 = 4 * 3 * 12 * 9 * 26 * 9 * 9 * 9
    
    for main_key, sub_dict in transitions.items():
        phase, regime, center = main_key
        base_val = phase * m_phase + regime * m_regime + center * m_center
        for sub_key, p_arr in sub_dict.items():
            c3, c7, o3, f3, r3, p3 = sub_key
            key_val = base_val + c3 * m_c3 + c7 * m_c7 + o3 * m_o3 + f3 * m_f3 + r3 * m_r3 + p3 * m_p3
            keys.append(key_val)
            probs.append(p_arr)
            
    keys = np.array(keys, dtype=np.int64)
    probs = np.array(probs, dtype=np.float32)
    
    # Sort keys for binary search
    sort_idx = np.argsort(keys)
    keys = keys[sort_idx]
    probs = probs[sort_idx]
    
    # Add a fallback key at index 0 for "unknown state"
    # We will map unknown states to index 0. The fallback probability will be "stay same" 
    # But since we don't know the center state easily without re-decoding, 
    # we can actually just make the fallback probabilities uniform, or we handle fallback specially.
    # Let's insert a dummy at index 0
    fallback_key = np.array([-1], dtype=np.int64)
    fallback_prob = np.zeros((1, 12), dtype=np.float32)
    fallback_prob[0, 0] = 1.0 # arbitrary
    
    keys = np.concatenate([fallback_key, keys])
    probs = np.concatenate([fallback_prob, probs])
    
    torch.save({
        'keys': torch.tensor(keys),
        'probs': torch.tensor(probs),
        'multipliers': (m_phase, m_regime, m_center, m_c3, m_c7, m_o3, m_f3, m_r3, m_p3)
    }, 'tasks/astar-island/gpu_search_tensor.pt')
    
    print(f"Compiled {len(keys)-1} unique transition states to gpu_search_tensor.pt")
    
if __name__ == '__main__':
    compile()
