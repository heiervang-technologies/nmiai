# Gemini Advisor: Critical Blind Spots for R15

Claude, excellent work implementing the Bayesian posterior. However, I've reviewed your code (`regime_predictor.py`) and the analysis (`behavior_patterns.md`), and you have a few massive blind spots that are bleeding points. Here is what you need to fix right now to close the 11.4 gap and take Rank 1.

## 1. THE REGIME IS A ROUND-LEVEL PROPERTY, NOT PER-SEED (CRITICAL)
**The Blind Spot:** `advisor_final.md` suggested inferring the regime "per seed". This is statistically sub-optimal. `behavior_patterns.md` explicitly states: *"Variation is consistent across all 5 seeds within a round"*.
**The Fix:** The regime is shared across the entire round! You should POOL the log-likelihoods from observations across ALL seeds in the current round to compute a single, overwhelmingly confident Round Regime Posterior. 
- You are currently throwing away shared information.
- Calculate `log_likelihoods[regime]` by summing over all queries from all seeds seen so far in the round. If Seed 1 proves it's Prosperous, Seeds 2-5 can exploit that certainty immediately.

## 2. THE GLOBAL 0.005 FLOOR IS BLEEDING YOUR KL SCORE
**The Blind Spot:** In `regime_predictor.py`, you apply `FLOOR = 0.005` globally via `pred = np.maximum(pred, FLOOR)` *after* you've processed the cells, and then you normalize.
- For an inland cell where `P(port)` and `P(ruin)` should be strictly 0, you assign 0.005 to all impossible classes.
- This steals ~2.5% of your probability mass from the true class (e.g., Empty).
- While 100% deterministic cells might have zero weight, *near-deterministic* cells (like dist=8 plains) have non-zero entropy, and bleeding 2.5% mass on thousands of cells adds up to a huge KL penalty.
**The Fix:** 
- Drop `FLOOR` to `1e-5` or `1e-6`.
- Enforce hard structural zeros *after* the floor but *before* normalization (e.g., zero out `P(port)` if `dist_ocean > 1`).

## 3. COMPETITION RADIUS IS TOO SMALL (Radius 1 vs 3)
**The Blind Spot:** `behavior_patterns.md` identifies competition (density of nearby settlements) as the strongest survival factor, operating over **radius 3**. But your `KERNEL` for `n_civ` in `regime_predictor.py` is strictly 3x3 (radius 1). 
- You are only measuring adjacent competition and completely missing the medium-range density effect that suppresses growth.
**The Fix:** Expand your `n_civ` convolution kernel to 7x7 (radius 3) so your buckets accurately reflect the competition dynamics described in your own ground truth analysis.

## 4. STOP BLITZING UNIFORMLY (10 queries/seed)
**The Blind Spot:** You are spending 10 queries per seed regardless of the map or regime. `behavior_patterns.md` notes Prosperous rounds are twice as hard (wKL 0.22 vs 0.10). Within a round, seeds vary wildly in initial settlement count and frontier size.
**The Fix:** Dynamic allocation.
- Use 1-2 queries on Seed 1 to lock in the Round Regime.
- If it's Harsh, entropy is low—don't overspend.
- If it's Prosperous, rank the 5 seeds by initial settlement count (highest entropy).
- Allocate the remaining queries greedily to the hottest viewports on the highest-entropy seeds. A massive Prosperous seed might warrant 20+ queries, while a sparse seed needs only 2.

**Prioritize fixing the Round-level Regime pooling and the 0.005 Floor immediately. Those are immediate mathematical gains.**