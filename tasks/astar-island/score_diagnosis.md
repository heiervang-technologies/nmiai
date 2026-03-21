# Score Diagnosis — Where We Lose Points

## The Bottom Line

Our ceiling (perfect template) is wKL=0.059. Our actual out-of-sample is wKL=0.205.
**We leave 71% of the available score on the table due to regime mismatch.**

The leader at ~92 raw likely achieves wKL ~0.04-0.05, meaning they essentially match the in-sample ceiling. They have solved regime detection.

## Loss Breakdown by Cell Category (all 12 rounds)

| Category | % of total loss | Description |
|----------|----------------|-------------|
| **mid_range** | **46.3%** | Cells at dist 2-5 from civ. Wrong growth/decay rate. |
| **coastal_frontier** | **17.0%** | Coastal cells near civ. Wrong port/settlement probability. |
| **near_civ** | **14.1%** | Cells at dist 0-2. Wrong immediate growth rate. |
| **forest_near_civ** | **11.7%** | Forest cells near civ. Wrong forest-to-settlement conversion. |
| **far_range** | **6.7%** | Cells at dist 5-10. Wrong reach estimate. |
| **init_settlement** | **4.0%** | Initial settlements. Wrong survival rate. |
| init_port | 0.2% | Negligible |
| remote | 0.0% | Negligible |

## Key Finding: Mid-Range Cells Are the #1 Problem

**46% of our total loss** comes from cells at distance 2-5 from civilization. These are the frontier growth cells where the round-specific regime matters most:
- In prosperous rounds (R6, R11): these cells grow aggressively, but our pooled prior predicts moderate growth → massive under-prediction
- In harsh rounds (R3, R10): these cells barely grow, but our pooled prior predicts moderate growth → over-prediction

The pooled average is always wrong for mid-range cells because the true behavior is bimodal.

## Per-Round Analysis

| Round | OOS wKL | Ceiling | Gap | Top Loss | Regime |
|-------|---------|---------|-----|----------|--------|
| R1 | 0.199 | 0.053 | 0.146 | mid_range 58% | moderate |
| R2 | 0.218 | 0.048 | 0.171 | mid_range 59% | moderate-high |
| R3 | 0.112 | 0.022 | 0.090 | near_civ 35% | harsh (low growth) |
| R4 | 0.139 | 0.028 | 0.110 | mid_range 57% | moderate |
| R5 | 0.139 | 0.060 | 0.078 | mid_range 41% | moderate |
| R6 | 0.342 | 0.073 | 0.269 | mid_range 53% | **prosperous** (high growth) |
| R7 | 0.222 | 0.129 | 0.093 | near_civ 33% | **prosperous** |
| R8 | 0.129 | 0.016 | 0.112 | mid_range 48% | moderate-low |
| R9 | 0.177 | 0.036 | 0.140 | mid_range 63% | moderate-high |
| R10 | 0.095 | 0.020 | 0.075 | near_civ 39% | harsh |
| R11 | 0.386 | 0.051 | 0.335 | mid_range 57% | **prosperous** (worst round) |
| R12 | 0.306 | 0.169 | 0.137 | near_civ 42% | **prosperous** |

## Worst Rounds: R6, R11, R12 (Prosperous Regime)

Our worst scores (wKL > 0.30) all come from prosperous rounds where growth is much higher than the pooled average. The pooled prior massively under-predicts settlement expansion in these rounds.

- R11: wKL=0.386, ceiling=0.051 → **we leave 87% on the table**
- R6: wKL=0.342, ceiling=0.073 → **we leave 79% on the table**
- R12: wKL=0.306, ceiling=0.169 → ceiling is also high (hardest round even in-sample)

## Ceiling Scores (What We'd Get With Perfect Templates)

| Round | Ceiling wKL | Approx Score |
|-------|-------------|-------------|
| R8 | 0.016 | ~96 |
| R10 | 0.020 | ~95 |
| R3 | 0.022 | ~95 |
| R4 | 0.028 | ~94 |
| R9 | 0.036 | ~93 |
| R2 | 0.048 | ~91 |
| R11 | 0.051 | ~91 |
| R1 | 0.053 | ~91 |
| R5 | 0.060 | ~89 |
| R6 | 0.073 | ~87 |
| R7 | 0.129 | ~80 |
| R12 | 0.169 | ~75 |

**Mean ceiling = 0.059, which would score ~89 on average.** The leader at ~92 is beating even our in-sample ceiling on some rounds, suggesting they have a better model architecture, not just better regime detection.

## What Must Change to Win

1. **Solve regime detection** (recovers ~70% of the gap): Even crude 3-way classification (harsh/moderate/prosperous) would cut mid-range error in half.

2. **Fix coastal_frontier predictions** (17% of loss): Port/settlement probability at coast varies wildly by regime. Need regime-conditional coastal prior.

3. **Fix near_civ growth rate** (14% of loss): Cells adjacent to civilization have very different growth rates per regime. This is the second-most diagnostic signal.

4. **Accept R7/R12 are hard even in-sample**: Ceiling wKL of 0.129/0.169 means these rounds have genuinely hard-to-predict dynamics even with the right template.
