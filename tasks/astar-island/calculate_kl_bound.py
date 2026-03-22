import torch
import numpy as np

def calculate_expected_kl():
    print("Loading sparse lookup dictionary...")
    transitions = torch.load('tasks/astar-island/sparse_lookup_v2.pt', weights_only=False)
    
    simulated_wkls = []
    entropy_weights = []
    
    for main_key, sub_dict in transitions.items():
        for sub_key, p_emp in sub_dict.items():
            # Normalize to avoid float32 precision errors with numpy multinomial
            p_emp = p_emp.astype(np.float64)
            p_emp /= p_emp.sum()
            
            if np.max(p_emp) < 0.99: # Stochastic state
                
                # The GT is constructed by running 200 simulation traces
                # P_gt is the empirical distribution of those 200 samples
                samples = np.random.multinomial(200, p_emp)
                p_gt = samples / 200.0
                
                # Our infinite Monte Carlo approaches P_emp exactly
                p_gt_safe = np.maximum(p_gt, 1e-8)
                p_emp_safe = np.maximum(p_emp, 1e-8)
                
                # KL(GT || Prediction) -> KL(p_gt || p_emp)
                kl = np.sum(p_gt_safe * (np.log(p_gt_safe) - np.log(p_emp_safe)))
                
                # Weighting factor (0.15 + entropy_bits)
                entropy_bits = -np.sum(p_gt_safe * np.log2(p_gt_safe))
                weight = 0.15 + entropy_bits
                
                simulated_wkls.append(weight * kl)
                entropy_weights.append(weight)
                
    mean_wkl = np.sum(simulated_wkls) / np.sum(entropy_weights)
    
    print(f"Simulated evaluation of Expected KL Divergence")
    print(f"Number of stochastic configurations analyzed: {len(simulated_wkls)}")
    print(f"Theoretical Evaluation Noise Floor (wKL): {mean_wkl:.5f}")
    
    # What if we only ran 1000 MC simulations instead of infinity?
    simulated_wkls_1000 = []
    for main_key, sub_dict in transitions.items():
        for sub_key, p_emp in sub_dict.items():
            p_emp = p_emp.astype(np.float64)
            p_emp /= p_emp.sum()
            if np.max(p_emp) < 0.99:
                p_gt = np.random.multinomial(200, p_emp) / 200.0
                p_pred = np.random.multinomial(1000, p_emp) / 1000.0
                
                p_gt_safe = np.maximum(p_gt, 1e-8)
                p_pred_safe = np.maximum(p_pred, 1e-8)
                
                kl = np.sum(p_gt_safe * (np.log(p_gt_safe) - np.log(p_pred_safe)))
                entropy_bits = -np.sum(p_gt_safe * np.log2(p_gt_safe))
                weight = 0.15 + entropy_bits
                simulated_wkls_1000.append(weight * kl)
                
    mean_wkl_1000 = np.sum(simulated_wkls_1000) / np.sum(entropy_weights)
    print(f"Expected score if we only run 1000 MC samples: {mean_wkl_1000:.5f}")

if __name__ == '__main__':
    calculate_expected_kl()
