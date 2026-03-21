# Parametric Decay Predictor Experiment

## Hypothesis

Replacing bucket-based lookup tables with continuous parametric decay functions
(`A * exp(-B * d) + C`) will eliminate bucket boundary artifacts and improve
prediction accuracy, especially at fractional distances.

## Method

For each (regime, cell_type, coast_status) group:

1. Collected `(distance_to_nearest_civ, P(class))` for every dynamic cell across
   all ground truth rounds in that regime
2. Binned at 0.5-unit distance resolution for stable fitting
3. Fitted `P(class|d) = A * exp(-B * d) + C` using `scipy.optimize.curve_fit`
   with sqrt(n)-weighted least squares
4. Used hierarchical fallback: fine (regime, ct, coast, n_ocean) -> medium
   (regime, ct, coast) -> coarse (regime, ct)
5. Structural constraints enforced: port=0 if not coastal, mountain=0 on non-mountain,
   ruin=0 at d>10

Evaluation: leave-one-round-out cross-validation on 15 rounds x 5 seeds = 75 maps.

## Results

### CV Comparison (mixture regime weights)

| Predictor | CV mean wKL | CV mean KL | Improvement |
|-----------|-------------|------------|-------------|
| regime_predictor (bucket) | 0.118970 | 0.130501 | baseline |
| parametric_predictor | 0.111288 | 0.128026 | -6.5% wKL |

### Per-Round CV Breakdown

| Round | Bucket wKL | Parametric wKL | Delta |
|-------|-----------|---------------|-------|
| R1 | 0.069547 | 0.066290 | -4.7% |
| R2 | 0.102240 | 0.089375 | -12.6% |
| R3 | 0.112048 | 0.117756 | +5.1% |
| R4 | 0.042222 | 0.045172 | +7.0% |
| R5 | 0.061480 | 0.057866 | -5.9% |
| R6 | 0.226442 | 0.198002 | -12.6% |
| R7 | 0.159422 | 0.148189 | -7.0% |
| R8 | 0.061579 | 0.067172 | +9.1% |
| R9 | 0.056816 | 0.050898 | -10.4% |
| R10 | 0.091075 | 0.097317 | +6.9% |
| R11 | 0.208849 | 0.183223 | -12.3% |
| R12 | 0.245185 | 0.231713 | -5.5% |
| R13 | 0.045721 | 0.046696 | +2.1% |
| R14 | 0.223691 | 0.203313 | -9.1% |
| R15 | 0.078234 | 0.066340 | -15.2% |

### In-Sample by Regime

| Regime | Bucket wKL | Parametric wKL |
|--------|-----------|---------------|
| mixture | 0.115478 | 0.107973 |

## Key Fitted Decay Parameters

### Prosperous Regime, Plains (non-coastal)

```
P(settlement|d) = +0.5285 * exp(-0.254 * d) + 0.0103
P(empty|d)      = -0.6772 * exp(-0.254 * d) + 0.9884
P(forest|d)     = +0.1028 * exp(-0.266 * d) + 0.0003
P(ruin|d)       = +0.0460 * exp(-0.225 * d) + 0.0010
```

### Moderate Regime, Plains (non-coastal)

```
P(settlement|d) = +0.3709 * exp(-0.375 * d) + 0.0034
P(empty|d)      = -0.4841 * exp(-0.358 * d) + 0.9963
P(forest|d)     = +0.0878 * exp(-0.326 * d) + 0.0001
P(ruin|d)       = +0.0271 * exp(-0.280 * d) + 0.0002
```

### Harsh Regime, Plains (non-coastal)

```
P(settlement|d) = +0.0543 * exp(-0.598 * d) + 0.0000
P(empty|d)      = -0.1877 * exp(-0.746 * d) + 1.0000
P(forest|d)     = +0.1348 * exp(-0.888 * d) + 0.0000
P(ruin|d)       = +0.0085 * exp(-0.596 * d) + 0.0000
```

## Interpretation

- **Prosperous**: Settlement amplitude 0.53, slow decay rate 0.25 -> half-life ~2.7 cells
- **Moderate**: Settlement amplitude 0.37, faster decay 0.38 -> half-life ~1.8 cells
- **Harsh**: Settlement amplitude 0.05, very fast decay 0.60 -> half-life ~1.2 cells
- Coast splits port probability from settlement probability correctly
- Parametric wins biggest on prosperous/moderate rounds where the smooth
  decay captures nuance the integer-distance buckets miss
- Harsh rounds slightly worse: very few data points per regime make curve
  fitting less stable than bucket averaging

## Round Regime Classification

| Regime | Rounds |
|--------|--------|
| Harsh | 3, 8, 10 |
| Moderate | 1, 2, 4, 5, 7, 9, 12, 13, 15 |
| Prosperous | 6, 11, 14 |

## Conclusion

The parametric predictor improves CV wKL by 6.5% over the bucket-based
regime_predictor. The continuous decay functions eliminate boundary artifacts
and provide better generalization, particularly on prosperous and moderate
rounds. The hierarchical grouping (fine/medium/coarse) ensures sufficient
data support at each level while retaining coast/ocean-neighbor specificity.
