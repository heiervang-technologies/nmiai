# Astar Island: Regime Prediction Analysis

## Executive Summary

Score variance (41-83 raw) is driven entirely by **regime mismatch** between the predictor's pooled average and the round's actual behavior. The regime (harsh/moderate/prosperous) is determined by hidden round-level simulation parameters, NOT by the initial grid. No initial grid feature correlates with regime (r < 0.15 across 60 data points). Observations are the only way to detect regime.

## Score Formula (Confirmed)

```
weighted_kl = sum(entropy_i * KL_i) / sum(entropy_i)
score = 100 * exp(-3 * weighted_kl)
```

Entropy weighting means high-uncertainty cells (frontier zone) dominate the score. Static cells (ocean, far inland) contribute almost nothing.

## The Three Regimes

| Regime | Rounds | Sett Survival | Frontier C1 | Frontier C0 | Description |
|--------|--------|---------------|-------------|-------------|-------------|
| HARSH | R3, R8, R10 | < 0.10 | 0.02 | 0.72 | Settlements die, frontier stays empty |
| MODERATE | R4, R9 | 0.22-0.27 | 0.13 | 0.61 | Some expansion, some collapse |
| PROSPEROUS | R1,R2,R5,R6,R7,R11,R12 | 0.33-0.60 | 0.24 | 0.53 | Settlements survive and expand |

## Critical Finding: Regime is Round-Dependent, NOT Grid-Dependent

Within each round, different seeds have completely different grids but nearly identical settlement survival rates:

| Round | Regime | Sett Surv Mean | Sett Surv Std | Range |
|-------|--------|----------------|---------------|-------|
| R3 | HARSH | 0.018 | 0.003 | 0.014-0.023 |
| R8 | HARSH | 0.067 | 0.010 | 0.054-0.083 |
| R10 | HARSH | 0.059 | 0.004 | 0.053-0.062 |
| R4 | MODERATE | 0.235 | 0.007 | 0.225-0.243 |
| R9 | MODERATE | 0.274 | 0.016 | 0.258-0.299 |
| R1 | PROSPEROUS | 0.416 | 0.025 | 0.373-0.440 |
| R11 | PROSPEROUS | 0.491 | 0.016 | 0.476-0.520 |
| R12 | PROSPEROUS | 0.595 | 0.032 | 0.554-0.636 |

Standard deviation within rounds (0.003-0.032) is an order of magnitude smaller than the range across rounds (0.018-0.595). The simulation's hidden parameters (growth rate, conflict intensity, winter severity) are set per-round, not per-grid.

## Grid Features Have ZERO Predictive Power

Tested across all 60 seeds (12 rounds x 5 seeds):

| Feature | Correlation with Sett Survival |
|---------|-------------------------------|
| n_civ | -0.036 |
| n_settlement | -0.027 |
| n_port | -0.067 |
| ocean_frac | -0.041 |
| mountain_frac | 0.135 |
| forest_frac | 0.027 |
| mean_dist_to_civ | -0.031 |

None even approach significance. The initial grid is randomly generated and has no bearing on the simulation regime.

## Why R9 and R11 Score High (81-83)

R9 (moderate regime, sett_surv=0.258) is the closest round to the cross-round average behavior:
- Cross-round average sett_surv = 0.316
- R9 distance from average = 0.096 (lowest of all rounds)
- The pooled predictor naturally fits moderate regimes best

R11 (prosperous, sett_surv=0.477) scores 82.6 despite being further from average (distance=0.528). This suggests either:
1. The template predictor detects R11 as in-sample (fingerprint match), or
2. R11's specific frontier patterns happen to align with the predictor's templates

## Why R12 Scores Low (43.5)

R12 is the most prosperous round (sett_surv=0.572), furthest from the pooled average. The predictor fails because:

1. **Frontier mismatch**: GT has C1=0.200 but predictor gives much less settlement probability to frontier cells
2. **The 0.01 floor hurts**: For cells that become settlement with 80%+ probability, the predictor assigns only 15-20%, wasting probability mass on C0
3. **Entropy weighting amplifies this**: 90% of R12's entropy-weighted loss is in the frontier zone (dist 1-3 from initial civ)
4. **Specific worst cells**: Plains/forest cells at dist 1-2 from settlements that become settlement with 80%+ probability, but predictor expects them to stay empty

Weighted KL comparison:
- R9: 0.069 -> score 81.4
- R11: 0.064 -> score 82.6
- R12: 0.277 -> score 43.5 (4x worse KL)

## Entropy-Weighted KL Loss by Zone

| Round | Score | Settlements | Frontier | Mid-range | Far |
|-------|-------|-------------|----------|-----------|-----|
| R9 | 81.4 | 5.5% | 59.2% | 33.5% | 1.6% |
| R11 | 82.6 | 2.8% | 56.2% | 38.2% | 2.8% |
| R12 | 43.5 | 9.7% | 89.9% | 0.0% | 0.0% |

Frontier cells (dist 1-3 from initial civ) dominate the loss in all rounds, but R12's loss is almost entirely frontier because those cells have the highest entropy AND the highest KL.

## Behavioral Distance Matrix (Selected)

How similar are rounds to each other (lower = more similar):

```
       R3    R9    R11   R12
R3   0.00  0.58  1.19  0.99
R9   0.58  0.00  0.61  0.43
R11  1.19  0.61  0.00  0.39
R12  0.99  0.43  0.39  0.00
```

R12 is most similar to R7 (0.18) and R1/R2 (0.21), but the pooled predictor dilutes this with harsh rounds (R3/R8/R10), which are 0.86-0.99 distant.

## Actionable Recommendations

### 1. Early Regime Detection via Observations (HIGH IMPACT)

Since regime cannot be predicted from the grid, use 2-3 queries per seed specifically for regime detection:
- Query a 3x3 viewport centered on an initial settlement cell
- If the settlement cell shows class 1 (settlement) in >40% of observed outcomes -> PROSPEROUS
- If class 0 (empty) dominates -> HARSH
- This detection can be done with a single query per seed

### 2. Regime-Specific Templates (HIGH IMPACT)

Instead of uniform mixture, maintain 3 regime-specific template averages:
- HARSH template: average of R3/R8/R10 patterns
- MODERATE template: average of R4/R9 patterns
- PROSPEROUS template: average of R1/R2/R5/R6/R7/R11/R12 patterns

After regime detection, use the matching template. This could close the gap from 43->70+ for prosperous rounds.

### 3. Optimize Query Budget Allocation

- Spend 2-3 queries per seed (10-15 total) on regime detection (settlement cells)
- Spend remaining 35-40 queries on high-entropy frontier cells for local refinement
- Focus queries on frontier zone (dist 1-3 from civ) since that's 56-90% of the score

### 4. Sub-Regime Detection Within Prosperous

The PROSPEROUS regime spans sett_surv 0.33-0.60. Within this range:
- LOW prosperous (R5: 0.34) vs HIGH prosperous (R12: 0.57)
- A few more observations could distinguish these sub-regimes
- Match to closest historical round template

### 5. Observation Priority for Settlement Cells

Initial settlement cells are the most diagnostic:
- In HARSH: they show class 0 (dead) = 0.70+
- In MODERATE: class 0 ~ 0.48, class 1 ~ 0.26
- In PROSPEROUS: class 1 (alive) = 0.33-0.57

A single settlement observation immediately constrains the regime.
