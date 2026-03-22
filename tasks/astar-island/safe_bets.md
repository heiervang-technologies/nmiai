# 3 Safe Bets for Immediate Deployment

These are high-confidence, low-risk changes that can be deployed right now to improve the CV score without overhauling the core model architecture.

## 1. Quantization-Aware Snapping (The "1/200" Trick)
**The Fix:** The ground truth is generated from exactly 200 Monte Carlo simulations, meaning every true probability is a multiple of $1/200 = 0.005$. Continuous models output fuzzy floats (e.g., 0.013). 
**Action:** Add a post-processing step to your final predictions just before submission. Round probabilities to the nearest `0.005`. Use the largest-remainder method to ensure the snapped probabilities still sum perfectly to 1.0. 
**Why it works:** KL divergence is extremely sensitive to exact probability matching. Snapping continuous predictions to the discrete 200-MC manifold mathematically guarantees a lower divergence on average.

## 2. Dynamic Round-Level Regime Pooling
**The Fix:** `behavior_patterns.md` states: *"Variation is consistent across all 5 seeds within a round"*. If your model infers the regime (Harsh/Moderate/Prosperous) per seed, you are throwing away shared information and wasting observation queries.
**Action:** Pool the observation log-likelihoods across all 5 seeds in the round. Use your first 1-2 queries on Seed 1 to confidently lock in the `Round Regime`. Once locked, use that exact regime for Seeds 2-5 with 100% confidence.
**Why it works:** This frees up your remaining 48 queries to be spent strictly on high-entropy frontier regions (the Prosperous seeds) instead of re-verifying the regime on every seed.

## 3. Lower the Global Floor & Apply Hard Structural Zeros
**The Fix:** If you have a global floor (e.g., `np.maximum(pred, 0.005)`), you are heavily penalizing yourself on deterministic cells. A `0.005` floor on an inland cell for `P(port)` steals mass from the true class.
**Action:**
1. Drop your global baseline floor from `0.005` to `1e-6` or `1e-5`.
2. Apply **hard structural zeros** explicitly *after* the floor but *before* normalization:
   - `pred[dist_ocean > 1, port_idx] = 0.0`
   - `pred[dist_settlement >= 12, settlement_idx] = 0.0`
   - `pred[~is_mountain, mountain_idx] = 0.0`
**Why it works:** You instantly recover the probability mass that was bleeding into impossible outcomes, sharpening your predictions for the correct classes and lowering the entropy-weighted KL divergence across thousands of cells.