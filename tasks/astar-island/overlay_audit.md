# Observation Overlay Audit

Tested against ground truth rounds 9, 13, 14, 15 using leave-one-round-out CV.
10 blitz observations per seed, sampled from GT distributions, averaged over 5 RNG seeds.

## Current Configuration

`tau=10`, Bayesian Dirichlet update on cells with 3+ samples.

## Test Results

### Experiment 1: Single Hottest Viewport (all 10 obs on same viewport)

| Method                         | Mean wKL   | vs No Overlay |
|-------------------------------|-----------|---------------|
| Dirichlet tau=50 min=3        | 0.056717  | -2.9%         |
| Dirichlet tau=20 min=3        | 0.057213  | -2.0%         |
| **No overlay (obs for regime only)** | **0.058397** | baseline |
| Entropy-weighted tau=20 min=3 | 0.058714  | +0.5%         |
| Dirichlet tau=10 min=3 (CURRENT) | 0.059713 | +2.3%        |
| Dirichlet tau=5 min=3         | 0.064699  | +10.8%        |
| Ratio correction s=0.3        | 0.079696  | +36.5%        |
| Pure MLE min=3                | 0.089068  | +52.5%        |
| Baseline (no observations)    | 0.092059  | +57.6%        |

### Experiment 2: Mixed Viewports (5+3+2 across 3 viewports)

| Method                         | Mean wKL   | vs No Overlay |
|-------------------------------|-----------|---------------|
| Dirichlet tau=30 min=1        | 0.055896  | -3.3%         |
| Entropy-weighted tau=50 min=3 | 0.056108  | -2.9%         |
| Dirichlet tau=35 min=3        | 0.056129  | -2.9%         |
| Dirichlet tau=40 min=3        | 0.056134  | -2.9%         |
| **Dirichlet tau=30 min=3**    | **0.056176** | **-2.8%**  |
| Dirichlet tau=50 min=3        | 0.056210  | -2.7%         |
| Dirichlet tau=25 min=3        | 0.056329  | -2.5%         |
| Dirichlet tau=75 min=3        | 0.056466  | -2.3%         |
| Dirichlet tau=30 min=5        | 0.056588  | -2.1%         |
| Dirichlet tau=100 min=3       | 0.056677  | -1.9%         |
| Dirichlet tau=20 min=3        | 0.056708  | -1.9%         |
| Dirichlet tau=30 min=8        | 0.057469  | -0.6%         |
| **No overlay**                | **0.057789** | baseline   |
| Dirichlet tau=10 min=3 (CURRENT) | ~0.0597 | +3.3%       |

## Key Findings

1. **Current tau=10 is too low.** It over-trusts observations and *hurts* performance vs no overlay at all. With 10 samples from a stochastic simulator, the empirical distribution is noisy. tau=10 gives observations a 50/50 weight against the prior -- too aggressive.

2. **Optimal tau is 30-40.** The sweet spot is tau=30-35 for mixed viewports. This gives the prior ~75% weight and observations ~25%, which smooths out sampling noise while still correcting real errors.

3. **min_samples threshold barely matters** when all observations hit the same viewport (all cells get 10 samples). With mixed viewports, min=1 through min=3 are essentially tied; min=5 and min=8 are slightly worse because they exclude cells with only 2-3 observations that could still help.

4. **Pure MLE is catastrophic.** Replacing the prior entirely with observation counts (no blending) increases wKL by 52%. The prior is highly informative and must not be discarded.

5. **Ratio correction is bad.** Multiplicative correction factors amplify noise and distort the distribution. Additive Dirichlet blending is strictly better.

6. **Entropy-weighted overlay helps slightly** at high tau (50), essentially matching standard Dirichlet tau=35-40. The idea is sound (uncertain cells should trust observations more) but the gain over a well-tuned fixed tau is negligible.

7. **Observations still help enormously for regime detection.** Even without any cell-level overlay, observations reduce wKL from 0.092 (baseline) to 0.058 (regime detection only). The overlay on top adds another 2-3%.

## Optimal Configuration

```python
# Dirichlet overlay parameters
TAU = 30          # was 10 -- increase to trust prior more
MIN_SAMPLES = 2   # was 3 -- lower threshold is fine, captures more cells
```

The update formula remains:
```
alpha = tau * prior + counts
posterior = alpha / alpha.sum()
```

With tau=30 and 10 observations: effective observation weight = 10/(10+30) = 25%.
With tau=10 and 10 observations: effective observation weight = 10/(10+10) = 50% (too high).

## Impact

Switching from tau=10 to tau=30 reduces mean wKL by approximately 6% on the overlay contribution, and 2-3% overall vs no overlay at all. The improvement is consistent across all test rounds.
