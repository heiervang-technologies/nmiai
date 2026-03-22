import numpy as np

def naive_snap(pred, target_sum=200):
    counts = np.floor(pred * target_sum).astype(int)
    remainders = (pred * target_sum) - counts
    deficiency = target_sum - counts.sum(axis=-1)
    
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if deficiency[i, j] > 0:
                d = int(deficiency[i, j])
                idx = np.argsort(remainders[i, j])[-d:]
                counts[i, j, idx] += 1
                
    snapped = counts / float(target_sum)
    snapped = np.maximum(snapped, 1e-6)
    snapped /= snapped.sum(axis=-1, keepdims=True)
    return snapped

def expected_value_snap(pred, target_sum=200, epsilon=1e-5):
    """
    Smarter expected-value snap.
    Instead of rounding to the nearest 1/200, we consider the binomial 
    distribution and the expected cross-entropy.
    """
    # GT is empirical counts from 200 trials with true probabilities 'pred' (assuming pred is perfectly accurate)
    # E_K[ - K/N * log(Q) ] = - P * log(Q)
    # The optimal Q is just P. 
    # But wait, if pred is our noisy MC estimate (e.g. from 20,000 sims), 
    # maybe we want to use Beta-Binomial or something?
    # Or maybe we just round, but carefully blend?
    
    # Let's try Bayesian smoothing of the snapped counts:
    counts = np.floor(pred * target_sum).astype(int)
    remainders = (pred * target_sum) - counts
    deficiency = target_sum - counts.sum(axis=-1)
    
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if deficiency[i, j] > 0:
                d = int(deficiency[i, j])
                idx = np.argsort(remainders[i, j])[-d:]
                counts[i, j, idx] += 1
                
    # Smarter snap: We use the quantized counts but add a Dirichlet prior 
    # based on the continuous prediction to prevent hard zeros
    snapped = (counts + epsilon) / (target_sum + pred.shape[-1] * epsilon)
    snapped /= snapped.sum(axis=-1, keepdims=True)
    return snapped

def test():
    # simulate some true underlying probabilities
    true_p = np.random.dirichlet(np.ones(6), size=(40, 40))
    
    # GT is 200 trials drawn from true_p
    gt_counts = np.zeros((40, 40, 6))
    for i in range(40):
        for j in range(40):
            gt_counts[i,j] = np.random.multinomial(200, true_p[i,j])
            
    gt_p = gt_counts / 200.0
    
    # We estimate pred from 20000 trials (MC sims)
    pred_counts = np.zeros((40, 40, 6))
    for i in range(40):
        for j in range(40):
            pred_counts[i,j] = np.random.multinomial(20000, true_p[i,j])
    pred_p = pred_counts / 20000.0
    
    # evaluate wKL
    def calc_kl(gt, p):
        gt = np.maximum(gt, 1e-12)
        p = np.maximum(p, 1e-12)
        return np.sum(gt * np.log(gt / p)) / (40 * 40)
        
    print("KL(GT || raw continuous pred_p):", calc_kl(gt_p, pred_p))
    print("KL(GT || naive_snap(pred_p)):   ", calc_kl(gt_p, naive_snap(pred_p)))
    print("KL(GT || EV_snap(pred_p)):      ", calc_kl(gt_p, expected_value_snap(pred_p)))

if __name__ == '__main__':
    for _ in range(5):
        test()
        print("-")
