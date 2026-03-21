# Eval Final — Regime Predictor Benchmark

## In-Sample Results (14 rounds, 70 seeds, no observations)

| Metric | Value |
|--------|-------|
| Mean KL | 0.132 |
| Mean wKL | 0.122 |
| Rounds | 14 |
| Seeds | 70 |

This is the **default-mix** score (no observations, regime weights {harsh: 0.23, moderate: 0.54, prosperous: 0.23}).

## Tau Observation Overlay — Leave-One-Round-Out CV

5 simulated hotspot observations per seed. Regime detection always active.

| Tau | Mean wKL | vs no-overlay | Notes |
|-----|----------|---------------|-------|
| 0 (no overlay) | **0.0769** | baseline | Regime detection only |
| 2 | 0.1102 | +43% WORSE | Way too aggressive |
| 5 | 0.0815 | +6% worse | Still too aggressive |
| 10 | 0.0751 | -2.3% better | Marginal |
| 20 | **0.0749** | **-2.6% better** | Best tau |
| 50 | 0.0746 | -3.0% better | Nearly same as 20 |

**Key finding: tau=2 is catastrophic (+43% worse). Low tau corrupts the regime prior with noisy single-observation counts. tau=20 or higher is marginally better than no overlay, but the gain is tiny (2-3%).**

**Recommendation: Use tau=20 for a small safe improvement, or skip the overlay entirely. Regime detection is what matters, not local cell updates.**

## Per-Round Breakdown (tau=0, regime detection only)

### Best Rounds (regime detection works perfectly)

| Round | wKL | Regime | Why it works |
|-------|-----|--------|-------------|
| R10 | 0.019 | harsh | Very clear harsh signal (near-zero settlement) |
| R8 | 0.025 | harsh | Same — harsh is easiest to detect |
| R3 | 0.033 | harsh | Same |
| R13 | 0.035 | moderate | Close to training moderate rounds |
| R4 | 0.037 | moderate | Same |

### Worst Rounds (regime detection fails or regime is unusual)

| Round | wKL | Regime | Why it fails |
|-------|-----|--------|-------------|
| R12 | 0.258 | moderate | Unique dynamics — high near-civ loss, unlike other moderate rounds |
| R7 | 0.163 | moderate | High settlement + high coastal — atypical moderate |
| R6 | 0.116 | prosperous | Extreme growth, only 3 prosperous training rounds → sparse priors |
| R14 | 0.093 | prosperous | Similar — prosperous regime has fewest training examples |
| R11 | 0.069 | prosperous | Best of the prosperous rounds |

### Pattern Analysis

**Harsh rounds (R3, R8, R10): Always score well (wKL 0.019-0.033).** The harsh regime is the most distinctive — near-zero settlement growth is easy to detect and predict. 3 training rounds give stable priors.

**Moderate rounds (R1, R4, R5, R9, R13): Score well (wKL 0.035-0.059).** Most training data (7 rounds), regime detection works. Exception: R12 (0.258) and R7 (0.163) are moderate rounds with atypical dynamics.

**Prosperous rounds (R2, R6, R11, R14): Mixed (wKL 0.069-0.116).** Only 3 training rounds, so regime priors are sparser. Growth patterns vary more within the prosperous class.

## R12 Deep Dive (Our Worst Round)

R12 is classified as moderate (frontier settle rate = 0.136) but scores terribly (wKL = 0.258). Even in-sample ceiling is high (0.169). This round has:
- Very high near-civ and forest-near-civ loss (42% + 25% of total)
- Unusually high coastal_frontier per-cell loss (avg wKL = 0.689)
- Dynamics unlike any other moderate round

**R12 is genuinely hard, not just a regime detection failure.** Even perfect regime assignment only gets to ~0.17 wKL.

## Comparison to Other Predictors

All values are leave-one-round-out CV with 5 observations:

| Predictor | CV wKL | Notes |
|-----------|--------|-------|
| Neighborhood (pooled, no obs) | ~0.104 | Baseline |
| Old template + 5 obs | ~0.082 | Template Bayesian posterior |
| **Regime predictor + 5 obs** | **0.077** | Simple settlement counting |
| Regime + tau=20 overlay | 0.075 | Marginal overlay gain |
| Oracle regime (ceiling) | ~0.068 | Perfect regime assignment |

## Recommendations for Competition

1. **Use regime_predictor with 5 observations, tau=20 overlay** — best honest out-of-sample score (wKL = 0.075)
2. **Focus observations on frontier cells (dist 1.5-6)** — this is what drives regime detection
3. **Don't waste queries on deep ocean/mountain/remote cells** — zero diagnostic value
4. **Accept R12-like rounds will be bad** — some rounds have genuinely hard dynamics even with perfect priors
5. **tau=2 is NEVER correct** — it destroys the prior. If using overlay at all, use tau >= 10
