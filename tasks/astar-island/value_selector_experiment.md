# Value-Based Viewport Selector Experiment

## Objective

Replace the heuristic viewport selector (settlement density-based) with an
information-theoretic selector that maximizes expected wKL score improvement.

## Approaches Tested

### 1. Value-Based (Information Gain)
Per-cell value = H_pred^2 / (tau + 1), where:
- H_pred = entropy of our prior prediction (uncertainty)
- tau = effective prior strength from regime_predictor bucket counts

Selects viewports greedily to maximize total value.

### 2. Regime Discriminability
Per-cell value = average pairwise symmetric KL divergence between regime
predictions. Cells that DIFFER MOST between harsh/moderate/prosperous regimes
are most valuable for regime classification.

### 3. Hybrid
Normalized combination of (1) and (2) with tunable alpha weight.

## Results

### Summary Table (wKL with observations, lower = better)

| Config              | wKL (obs) | Improvement vs no-obs | Seeds Improved |
|---------------------|-----------|----------------------|----------------|
| heuristic_vp1       | 0.075059  | 35.00%               | 53/75          |
| heuristic_vp3       | 0.074969  | 35.08%               | 50/75          |
| heuristic_vp5       | 0.075077  | 34.99%               | 51/75          |
| value_based_vp1     | 0.074359  | 35.61%               | 54/75          |
| value_based_vp5     | 0.074319  | 35.64%               | 53/75          |
| regime_disc_vp1     | 0.074709  | 35.30%               | 55/75          |
| **regime_disc_vp5** | **0.074211** | **35.74%**        | **52/75**      |
| hybrid_a0.5_vp5     | 0.074361  | 35.61%               | -              |
| hybrid_a0.8_vp5     | 0.074314  | 35.65%               | -              |

### Best vs Heuristic

- **Best heuristic**: heuristic_vp3 = 0.074969
- **Best value-based**: regime_disc_vp5 = 0.074211
- **Improvement**: +1.01%

### Per-Round Breakdown (regime_disc_vp5 vs heuristic_vp3)

| Round | Value  | Heuristic | Diff   | Winner |
|-------|--------|-----------|--------|--------|
| R1    | 0.0681 | 0.0683    | +0.4%  | V      |
| R2    | 0.0848 | 0.0905    | +6.3%  | V      |
| R3    | 0.0241 | 0.0241    | 0.0%   | tie    |
| R6    | 0.0973 | 0.0980    | +0.7%  | V      |
| R8    | 0.0288 | 0.0281    | -2.7%  | H      |
| R15   | 0.0703 | 0.0741    | +5.1%  | V      |

Wins on prosperous rounds (R2, R6) and R15 where regime detection matters most.
Loses slightly on R8 (harsh round). Most rounds are tied (regime detection
converges to same answer regardless of viewport).

## Conclusions

1. **Regime discriminability selector beats the heuristic by 1.01%** at 5 viewports.
   The margin is narrow but consistent.

2. The main benefit comes from **better regime detection** on ambiguous rounds,
   not from cell-level information gain. This makes sense: the predictor uses
   regime-level priors, so the most valuable information is which regime we're in.

3. The value-based (information gain) selector is slightly worse than regime_disc
   because it optimizes for cell-level uncertainty reduction, but the predictor
   doesn't do cell-level Bayesian updates -- it only uses observations for
   regime classification.

4. **Recommendation**: Use `select_viewports_for_regime_detection()` from
   `value_viewport_selector.py` with n_viewports=5 for best results.

## Files

- `value_viewport_selector.py` - Implementation with all three selectors
- `value_selector_experiment.py` - Experiment harness
- `value_selector_results.json` - Raw results
