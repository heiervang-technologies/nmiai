# Bayesian Per-Round Template Predictor: Experiment Results

## Concept

Instead of grouping rounds into 3 regime buckets (harsh/moderate/prosperous),
treat **each historical round as its own template**. Weight templates by
observation likelihood. This captures within-regime variation that bucket
averages miss.

## Method

1. **Per-round bucket priors**: Same feature set as regime_predictor
   (initial_type, dist_to_civ, ocean_adj, n_ocean, n_civ, coast) but tables
   built per-round rather than per-regime.

2. **Bayesian weighting**: For each template round R_t, compute
   log P(observations | R_t's bucket priors). Posterior: w_t proportional to
   exp(log_lik_t) with uniform prior.

3. **Mixture prediction**: Weighted average of all template predictions.

4. **Observation overlay**: tau=20 Bayesian update on cells with 2+ observations.

5. **Honest CV**: Leave-one-round-out, 16 folds. Each fold trains on 15 other
   rounds. np.random.seed(42) for reproducible observation simulation.

## Results: Honest Leave-One-Round-Out CV

| Round | No Obs wKL | With Obs wKL | Delta | Status |
|-------|-----------|-------------|-------|--------|
| R1    | 0.058678  | 0.040031    | -0.018647 | BETTER |
| R2    | 0.091421  | 0.034759    | -0.056662 | BETTER |
| R3    | 0.110327  | 0.017824    | -0.092503 | BETTER |
| R4    | 0.024323  | 0.017525    | -0.006798 | BETTER |
| R5    | 0.046366  | 0.062874    | +0.016508 | WORSE  |
| R6    | 0.242848  | 0.084672    | -0.158176 | BETTER |
| R7    | 0.147651  | 0.139414    | -0.008237 | BETTER |
| R8    | 0.054857  | 0.026321    | -0.028536 | BETTER |
| R9    | 0.040471  | 0.023696    | -0.016775 | BETTER |
| R10   | 0.088043  | 0.022427    | -0.065616 | BETTER |
| R11   | 0.221901  | 0.059763    | -0.162139 | BETTER |
| R12   | 0.240584  | 0.160476    | -0.080108 | BETTER |
| R13   | 0.026602  | 0.021475    | -0.005127 | BETTER |
| R14   | 0.236500  | 0.073842    | -0.162659 | BETTER |
| R15   | 0.064292  | 0.023004    | -0.041288 | BETTER |
| R16   | 0.033727  | 0.030103    | -0.003624 | BETTER |

### Summary

| Metric | Value |
|--------|-------|
| Mean no_obs wKL  | **0.108037** |
| Mean with_obs wKL | **0.052388** |
| Improvement from obs | **+51.5%** |

## Comparison to Regime Predictor Baseline

| Predictor | No Obs wKL | With Obs wKL |
|-----------|-----------|-------------|
| Regime (3 buckets) | 0.119 | 0.070 |
| Bayesian Template (per-round) | **0.108** | **0.052** |

**Improvement over regime predictor**:
- No obs: 0.119 -> 0.108 = **9.2% better**
- With obs: 0.070 -> 0.052 = **25.7% better**

## Key Observations

1. **Observations help on 15/16 rounds** (only R5 regresses slightly).
   The Bayesian weighting reliably identifies the most similar historical round.

2. **Template selection is sensible**: R3 picks R10 (both harsh), R6 picks R11
   (both prosperous), R4 picks R13 (both moderate). The system discovers regime
   structure without explicit regime labels.

3. **Biggest wins on high-variance rounds**: R6, R11, R14 (prosperous rounds with
   high entropy) show 0.16+ wKL improvement from observations.

4. **R5 is the only regression** (+0.017): observations push toward R13 when
   R5 has its own unique dynamics. The 3-viewport budget may be insufficient for
   this case.

## Configuration

- 3 viewports, hotspot selection (highest entropy regions)
- tau=20 local observation overlay on cells with 2+ observations
- Hierarchical bucket fallback: fine -> mid -> coarse -> broad
- Shrinkage toward pooled: alpha = support / (support + 3.0)
- Floor: 1e-6, structural zeros enforced (no port off coast, no mountain on
  non-mountain, no ruin far from civ)
