# Astar Island Temporal Audit

Analysis of 75 ground truth files (rounds 1-15, seeds 0-4).

## Executive Summary

**Key exploitable findings:**
1. Regime is a ROUND-LEVEL parameter (not per-seed) -- all 5 seeds in a round behave identically
2. Mod-5 ANOVA is significant (p=0.047) -- position within 5-round cycle predicts regime
3. Position 0 (R1,R6,R11) is consistently the highest-growth round in each cycle
4. Position 2 (R3,R8,R13) is consistently the lowest-growth round in each cycle
5. Border cells (row/col 0 and 39) are ALWAYS ocean (class 0 with probability 1.0)
6. Mountains NEVER change -- P(mountain) = 1.0000 for all mountain cells in all rounds
7. Ocean cells NEVER change -- P(empty) = 1.0000 for all ocean cells
8. Initial grid composition does NOT predict regime -- the regime is a hidden simulator parameter
9. Settlement survival probability is a strong regime indicator queryable from a single cell

## 1. Regime Sequence

| Round | Growth Ratio | Regime | Entropy |
|-------|-------------|--------|---------|
| 1 | 6.20 | prosperous | 0.5538 |
| 2 | 6.29 | prosperous | 0.6919 |
| 3 | 0.09 | **harsh** | 0.0681 |
| 4 | 2.93 | moderate | 0.4654 |
| 5 | 4.03 | moderate | 0.4818 |
| 6 | 7.82 | prosperous | 0.8082 |
| 7 | 4.20 | moderate | 0.3917 |
| 8 | 0.80 | **stagnant** | 0.2697 |
| 9 | 4.77 | moderate | 0.6114 |
| 10 | 0.32 | **stagnant** | 0.1120 |
| 11 | 10.72 | **boom** | 0.6822 |
| 12 | 4.73 | moderate | 0.2687 |
| 13 | 2.96 | moderate | 0.5082 |
| 14 | 8.98 | **boom** | 0.6770 |
| 15 | 6.40 | prosperous | 0.6662 |

Growth ratio = (final settlements + ports) / initial settlements.

## 2. Regime Transition Matrix

No strong pattern in what regime follows what. The transition matrix is roughly uniform -- knowing the current regime does NOT predict the next one.

Autocorrelation at lag 1: r = -0.185 (weak negative, not significant).

## 3. Mod-5 Cycle Pattern (SIGNIFICANT: p=0.047)

ANOVA by round_number mod 5 is significant. The 5-round cycle shows:

| Position (mod 5) | Rounds | Growth Values | Mean | Character |
|-------------------|--------|---------------|------|-----------|
| 0 | R1, R6, R11 | 6.2, 7.8, 10.7 | 8.25 | **Consistently HIGH** |
| 1 | R2, R7, R12 | 6.3, 4.2, 4.7 | 5.07 | Moderate |
| 2 | R3, R8, R13 | 0.1, 0.8, 3.0 | 1.28 | **Consistently LOW** |
| 3 | R4, R9, R14 | 2.9, 4.8, 9.0 | 5.56 | Rising |
| 4 | R5, R10, R15 | 4.0, 0.3, 6.4 | 3.58 | Variable |

**Prediction for Round 16:** Position 0 in cycle 4 -> expect HIGH growth (like R1=6.2, R6=7.8, R11=10.7). Predicted growth ~11+.

**Prediction for Round 17:** Position 1 -> moderate growth (~4-5).

**Prediction for Round 18:** Position 2 -> LOW growth (~3-4, trending up from 0.1->0.8->3.0).

Within each 5-round cycle, the rank ordering of positions is:
- Cycle 1: [2, 1, 5, 4, 3]
- Cycle 2: [1, 3, 4, 2, 5]
- Cycle 3: [1, 4, 5, 2, 3]

Position 0 (mod 5) is rank 1 or 2 in ALL cycles. Position 2 (mod 5) is rank 4 or 5 in ALL cycles.

## 4. Lag-5 Autocorrelation (r=0.55, p=0.08)

Nearly significant autocorrelation at lag 5. Growth in round N correlates with growth in round N+5:

| R(N) | R(N+5) |
|------|--------|
| R1 (6.2) | R6 (7.8) |
| R2 (6.3) | R7 (4.2) |
| R3 (0.1) | R8 (0.8) |
| R4 (2.9) | R9 (4.8) |
| R5 (4.0) | R10 (0.3) |
| R6 (7.8) | R11 (10.7) |
| R7 (4.2) | R12 (4.7) |
| R8 (0.8) | R13 (3.0) |
| R9 (4.8) | R14 (9.0) |
| R10 (0.3) | R15 (6.4) |

This is consistent with the mod-5 pattern above.

Consistent across all seeds (per-seed lag-5 r values: 0.50-0.70).

## 5. Escalation (Not Significant)

Within each mod-5 position, there may be an upward trend (mean escalation = +1.43 per cycle), but this is not statistically significant (p=0.15). With only 3 data points per position, we cannot confirm escalation.

## 6. Map Features -- What Is Constant

### Always constant:
- **Border cells** (row/col 0 and 39): ALWAYS ocean (code 10) in initial grid, ALWAYS P(empty)=1.0 in GT
- **Mountains**: ALWAYS stay mountain -- P(mountain)=1.0000 in every round and seed
- **Ocean cells**: ALWAYS stay empty -- P(empty)=1.0000 in every round and seed

### Not constant across rounds:
- Ocean positions CHANGE between rounds (even for same seed number)
- Mountain positions CHANGE between rounds
- All grids are independently generated -- no grid is reused
- ~650-750 cells differ between any two grids (out of 1600)

### Constant generator parameters:
- Ocean fraction: ~12-14% (stable across all rounds)
- Plains fraction: ~60-62%
- Forest fraction: ~21-22%
- Mountain fraction: ~1.5-3.5%
- Settlement fraction: ~2.5-3.2%
- Settlement adjacency: 0.000 -- initial settlements are NEVER placed adjacent to each other

No correlation between initial grid composition and round number (all p > 0.16).

## 7. Regime Is Not In The Initial Grid

The initial grid does NOT encode the regime. Tested: ocean count, plains count, forest count, mountain count, settlement count -- NONE differ significantly between harsh and prosperous rounds (all p > 0.5).

The regime is a hidden simulation parameter that affects dynamics (growth rate, conflict intensity, winter harshness) but is not visible in the initial state.

## 8. Cross-Seed Consistency (IMPORTANT)

Within each round, all 5 seeds produce very similar growth ratios:
- CV (coefficient of variation) ranges from 0.06 to 0.23
- Example: Round 3 harsh regime -- ALL 5 seeds have growth ~0.1
- Example: Round 11 boom -- ALL 5 seeds have growth 9.2-11.3

**Implication:** Once you identify the regime from ONE seed's query, you know the regime for ALL 5 seeds in that round. This means you can allocate more queries to observation rather than regime classification.

## 9. Settlement Survival as Regime Classifier

P(still settlement | initially settlement) is a strong regime indicator:

| Regime | Survival Probability |
|--------|---------------------|
| Harsh (R3) | 0.018 |
| Stagnant (R8, R10) | 0.059-0.067 |
| Moderate (R4, R5, R9, R13) | 0.228-0.330 |
| Prosperous (R1, R2, R6, R7, R15) | 0.326-0.423 |
| Boom (R11, R14) | 0.491-0.519 |
| High-mod (R12) | 0.595 |

**Strategy:** Query a known settlement cell early. If P(settlement) < 0.1 -> harsh. If > 0.4 -> prosperous/boom.

## 10. Transition Rates by Regime

### Harsh regime (R3):
- Settlement -> 68% empty, 2% settlement, 0% port, 0.4% ruin, 30% forest
- Forest -> 97% forest (barely changes)
- Plains -> 99% empty

### Prosperous regime (R6):
- Settlement -> 38% empty, 41% settlement, 0% port, 4% ruin, 17% forest
- Forest -> 15% empty, 25% settlement, 1.5% port, 3% ruin, 55% forest
- Plains -> 64% empty, 25% settlement, 2% port, 3% ruin, 6% forest

### Boom regime (R11):
- Settlement -> 32% empty, 49% settlement, 0.5% port, 4% ruin, 15% forest
- Forest -> 8% empty, 29% settlement, 2% port, 2% ruin, 60% forest
- Plains -> 66% empty, 28% settlement, 2% port, 2% ruin, 4% forest

## 11. Port and Ruin Patterns

- Port/settlement ratio is relatively stable (0.05-0.09) except in harsh rounds (0.02-0.03)
- Ruin/settlement ratio is highest in harsh rounds (0.22 in R3) because few settlements survive but some still generate ruins
- In prosperous rounds, ruin ratio is low (0.06-0.09) because settlements thrive

## 12. No Systematic Early vs Late Difference

- Early rounds (R1-7) mean final settlement: 208.3
- Late rounds (R8-15) mean final settlement: 208.0
- T-test: p=0.99 (completely identical)
- No entropy drift either (p=0.70)

## Actionable Recommendations

1. **For round 16 (next up, position 0 mod 5):** Use PROSPEROUS/BOOM priors. Expect growth ratio 8-12+. Allocate transition probabilities assuming heavy settlement expansion into forests and plains.

2. **Quick regime detection:** Spend 1 query per seed on a known settlement cell. P(settlement) immediately classifies the regime.

3. **Free predictions:** Mountains always stay mountain. Ocean/border always stay empty. These cells need zero queries.

4. **Regime-conditional priors:** Build separate transition probability tables for each regime level. The mod-5 position gives a prior on which regime to expect, and a single query confirms it.

5. **Round 16 specific:** Based on the pattern (position 0 = high growth), use boom-tier transition tables as your prior. The escalation trend within position 0 (6.2 -> 7.8 -> 10.7) suggests round 16 could be the highest growth yet.
