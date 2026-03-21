# Residual Audit: Regime Predictor Systematic Error Patterns

Analyzed 86,615 dynamic cells across 75 seeds, 15 rounds.
Overall mean wKL per dynamic cell: 0.112226 (in-sample), 0.119901 (CV)

---

## EXECUTIVE SUMMARY: Top 3 Fixes to Close the Gap

| Fix | Estimated wKL Improvement | Complexity |
|-----|--------------------------|------------|
| 1. Better regime detection from initial observations | 35% in-sample / 31% CV | Medium |
| 2. Systematic settlement under-prediction across ALL distances | ~10-15% | Easy |
| 3. Forest CC size and n_forest as bucket features | ~5% | Easy |

Current: 0.120 CV wKL. Oracle regime alone: 0.082 CV wKL. Gap to close: ~6 points on competition score.

---

## FINDING 1: REGIME MIXTURE IS THE #1 ERROR SOURCE (35% of total wKL)

The predictor uses a fixed mixture (harsh=23%, moderate=54%, prosperous=23%) when no observations are available. This BLEEDS regime priors into each other, causing massive systematic errors:

| Regime | Cells | Mean wKL | Settlement Residual | Direction |
|--------|-------|----------|--------------------|-----------|
| harsh | 11,752 | 0.076 | -0.136 | OVER-predict settle by 13.6% |
| moderate | 54,803 | 0.081 | +0.024 | Under-predict settle by 2.4% |
| prosperous | 20,060 | 0.219 | +0.139 | Under-predict settle by 13.9% |

**Oracle regime (knowing correct regime) reduces total wKL from 9,720 to 6,322 (35% improvement).**
**Cross-validated oracle reduces mean wKL from 0.120 to 0.082 (31% improvement).**

Prosperous rounds (R6, R11, R14) are the worst: mean wKL = 0.22 vs 0.08 for moderate.

### Round frontier rates (settlement expansion rates):

| Round | Frontier Rate | Regime | wKL |
|-------|--------------|--------|-----|
| R3 | 0.003 | harsh | 0.109 |
| R10 | 0.010 | harsh | 0.089 |
| R8 | 0.027 | harsh | 0.060 |
| R4 | 0.096 | moderate | 0.038 |
| R13 | 0.101 | moderate | 0.041 |
| R5 | 0.131 | moderate | 0.061 |
| R12 | 0.145 | moderate | **0.243** |
| R9 | 0.146 | moderate | 0.053 |
| R7 | 0.150 | moderate | **0.158** |
| R1 | 0.177 | moderate | 0.071 |
| R15 | 0.192 | moderate | 0.072 |
| R2 | 0.199 | moderate | 0.095 |
| R6 | 0.257 | prosperous | 0.223 |
| R14 | 0.275 | prosperous | 0.228 |
| R11 | 0.308 | prosperous | 0.204 |

R7 (wKL=0.158) and R12 (wKL=0.243) are moderate-classified but have anomalously high error. R12 especially has very few dynamic cells (649-789) and high settlement probability (~30%), suggesting a unique pattern not captured by the moderate prior.

### Fix: Use initial grid features to predict regime before any observations

The initial grid contains information about settlement density, number of civ cells, and map structure that correlates with regime. Early observations (even 1 blitz query) can dramatically narrow regime uncertainty via Bayesian update.

---

## FINDING 2: SYSTEMATIC SETTLEMENT UNDER-PREDICTION (+0.029 mean residual)

Across ALL cell types except Port, we consistently under-predict settlement probability:

| Cell Type | N | Settle Residual | Mean wKL |
|-----------|---|----------------|----------|
| Forest (F) | 21,790 | +0.029 | 0.138 |
| Land (L) | 61,387 | +0.029 | 0.100 |
| Settlement (S) | 3,309 | +0.011 | 0.176 |
| Port (P) | 129 | -0.010 | 0.205 |

This under-prediction is consistent at EVERY distance bin for forest cells:

| Dist to Civ | N | Actual Settle | Predicted Settle | Gap |
|-------------|---|--------------|-----------------|-----|
| [1.0,1.5) | 6,586 | 0.224 | 0.206 | +0.018 |
| [2.0,2.5) | 6,174 | 0.177 | 0.147 | +0.030 |
| [2.5,3.0) | 1,221 | 0.164 | 0.116 | +0.049 |
| [3.0,3.5) | 2,784 | 0.150 | 0.106 | +0.044 |
| [4.0,4.5) | 2,032 | 0.106 | 0.072 | +0.034 |
| [5.0,5.5) | 962 | 0.076 | 0.049 | +0.027 |
| [7.0,7.5) | 161 | 0.035 | 0.017 | +0.018 |

The gap is NOT random -- it grows from dist 1 to 3, then slowly shrinks. This suggests the decay curve is STEEPER in our model than reality. The actual settlement probability decays more slowly with distance than what our lookup tables predict.

### Fix: Apply a global correction factor that boosts settlement probability by ~0.025 and reduces ocean/forest correspondingly, OR fit a smooth decay curve instead of binned lookup.

---

## FINDING 3: FOREST CC SIZE EFFECT (r=-0.14 correlation with forest residual)

Larger forest patches have LOWER forest retention probability, but our model predicts the same for all CC sizes:

| CC Size | N | Mean wKL | Settle Resid | Forest Resid | Actual Forest Prob |
|---------|---|----------|-------------|-------------|-------------------|
| [1,5) | 15,629 | 0.136 | +0.026 | -0.029 | 0.718 |
| [5,15) | 5,350 | 0.139 | +0.035 | -0.036 | 0.709 |
| [15,50) | 811 | 0.165 | +0.054 | -0.060 | 0.689 |

Large forest CCs (15+ cells) lose 3% more forest prob and gain 5.4% more settlement than our model predicts. This makes physical sense: large forest patches present more opportunity for settlement expansion.

### Fix: Add forest_cc_size as a bucket feature (binned: 0, 1-4, 5-14, 15+), or apply a CC-size-dependent correction to forest cells.

---

## FINDING 4: n_forest CORRELATION (monotonic settlement under-prediction)

The number of forest neighbors (n_forest) correlates with settlement under-prediction:

| n_forest | N | Settle Resid | Forest Resid |
|----------|---|-------------|-------------|
| 0 | 11,489 | +0.018 | -0.001 |
| 3 | 15,428 | +0.035 | -0.003 |
| 5 | 1,978 | +0.042 | -0.012 |
| 8 | 12 | +0.115 | -0.138 |

More forest neighbors = more settlement under-prediction. Cells deep in forest are more likely to become settlements than our model predicts.

### Fix: Add n_forest to the bucket key, or use it as a correction factor.

---

## FINDING 5: HETEROGENEITY DRIVES PORT/MOUNTAIN RESIDUALS

The strongest correlations with residuals involve features we partly use but could use better:

| Feature | Strongest Residual Correlations |
|---------|-------------------------------|
| n_ocean | Port: +0.31, Mountain: -0.59 |
| heterogeneity | Port: +0.12, Mountain: -0.38, Settlement: -0.07 |
| dist_civ | Mountain: +0.40 |
| forest_cc_size | Forest: -0.14, Ocean: +0.08 |

The n_ocean -> Mountain correlation (-0.59) is the strongest in the entire table. This is an artifact: more ocean neighbors means the benchmark's 0.01 mountain floor matters less (fewer residual classes to compete with). Not directly actionable.

Heterogeneity (number of distinct terrain types in 3x3 neighborhood) is NOT used in the bucket key but correlates with wKL (0.078 at het=1 vs 0.170 at het=4 vs 0.233 at het=5).

### Fix: Consider adding heterogeneity bin (1-2, 3, 4+) to the bucket key.

---

## FINDING 6: COASTAL FOREST INTERACTION

Forest cells that are both coastal AND near civilization have a distinct pattern not fully captured:

| Condition | N | Mean wKL | Port Resid | Forest Resid |
|-----------|---|----------|-----------|-------------|
| Coast + Near Civ (<4) | 2,320 | 0.181 | +0.016 | -0.043 |
| Only Near Civ | 15,604 | 0.147 | -0.010 | -0.031 |
| Coast Only | 1,126 | 0.068 | +0.010 | -0.024 |
| Neither | 2,740 | 0.079 | -0.010 | -0.028 |

Coastal forest near civilization has the HIGHEST wKL (0.181) and we under-predict Port by +0.016. The coast flag IS in the bucket key, but the interaction with dist_civ could be more nuanced.

---

## FINDING 7: BORDER CELLS HAVE PORT RESIDUAL BIAS

| Location | N | Port Resid | Ocean Resid |
|----------|---|-----------|-------------|
| Near-border (2 rows) | 7,819 | +0.021 | -0.023 |
| Interior | 78,796 | -0.009 | -0.011 |

Border cells have +0.021 Port residual (we under-predict port) vs -0.009 for interior. This suggests edge effects in the simulation that we don't model.

---

## FINDING 8: MAP POSITION IS NOT SIGNIFICANT

Quadrant analysis shows negligible positional effects:

| Quadrant | Mean wKL | Settle Resid |
|----------|----------|-------------|
| Top-Left | 0.115 | +0.030 |
| Top-Right | 0.110 | +0.029 |
| Bottom-Left | 0.111 | +0.028 |
| Bottom-Right | 0.113 | +0.028 |

No map position bias detected. Not worth pursuing.

---

## FINDING 9: TOP 5% CELLS = 27% OF TOTAL wKL

4,331 cells (top 5% by wKL) contribute 2,638 wKL (27.1% of total). These are:
- 57% Land cells, 40% Forest cells
- Closer to civilization (mean dist_civ=1.95 vs 2.50 overall)
- More coastal (21.7% vs 16.3%)
- Higher heterogeneity (2.67 vs 2.46)
- Massive settlement under-prediction (+0.30 residual)

These worst cells are almost entirely in prosperous rounds where the mixture underestimates settlement growth.

---

## FINDING 10: PER-ROUND wKL BREAKDOWN

Worst rounds by mean wKL:
1. R12 (moderate): 0.243 -- anomalous high-growth moderate round
2. R14 (prosperous): 0.228
3. R6 (prosperous): 0.223
4. R11 (prosperous): 0.205
5. R7 (moderate): 0.158

Best rounds:
1. R4 (moderate): 0.038
2. R13 (moderate): 0.041
3. R9 (moderate): 0.053

The 3 prosperous rounds alone account for ~45% of total wKL despite being only 20% of rounds.

---

## PRIORITY ACTION ITEMS

1. **HIGHEST IMPACT**: Improve regime detection. Use initial grid features (settlement count, civ density, map structure) to produce a better prior than the flat 23/54/23 mixture. Even a rough classifier that identifies "definitely harsh" vs "definitely prosperous" would save ~30% of wKL.

2. **MEDIUM IMPACT**: Apply a settlement boost correction. We systematically under-predict settlement by +0.029 across all cell types. A simple post-hoc correction of +0.02 to settlement (redistributed from ocean/forest proportionally) would help.

3. **MEDIUM IMPACT**: Add forest_cc_size and n_forest to the bucket key. Large forest patches convert to settlement at higher rates than small ones.

4. **LOW IMPACT**: Add heterogeneity to the bucket key. High-heterogeneity cells have 2x the wKL of low-heterogeneity cells.

5. **LOW IMPACT**: Smooth the distance decay curve. The settlement probability drops too steeply between integer distance bins, especially at dist 2-4 where the gap is largest (+0.044-0.049).

---

## RAW DATA TABLES

### Per-Class Residual Bias by Cell Type

Positive residual = we UNDER-predict (GT > pred). Negative = we OVER-predict.

#### Cell type: S (3309 cells, mean wKL=0.176173)
| Class | Mean Residual | Mean |Residual| | Direction |
|-------|--------------|----------------|----------|
| Ocean | +0.00449 | 0.11363 | UNDER |
| Settlement | +0.01143 | 0.15724 | UNDER |
| Port | -0.00903 | 0.01325 | OVER |
| Ruin | +0.00059 | 0.01408 | ~OK |
| Forest | +0.00232 | 0.06118 | UNDER |
| Mountain | -0.00981 | 0.00981 | OVER |

#### Cell type: P (129 cells, mean wKL=0.204608)
| Class | Mean Residual | Mean |Residual| | Direction |
|-------|--------------|----------------|----------|
| Ocean | +0.01548 | 0.11817 | UNDER |
| Settlement | -0.00975 | 0.05904 | OVER |
| Port | -0.00374 | 0.11238 | OVER |
| Ruin | -0.00129 | 0.01220 | OVER |
| Forest | +0.00920 | 0.06360 | UNDER |
| Mountain | -0.00990 | 0.00990 | OVER |

#### Cell type: F (21790 cells, mean wKL=0.137704)
| Class | Mean Residual | Mean |Residual| | Direction |
|-------|--------------|----------------|----------|
| Ocean | +0.01626 | 0.04556 | UNDER |
| Settlement | +0.02942 | 0.09581 | UNDER |
| Port | -0.00605 | 0.01547 | OVER |
| Ruin | +0.00197 | 0.01074 | UNDER |
| Forest | -0.03178 | 0.13559 | OVER |
| Mountain | -0.00981 | 0.00981 | OVER |

#### Cell type: L (61387 cells, mean wKL=0.099541)
| Class | Mean Residual | Mean |Residual| | Direction |
|-------|--------------|----------------|----------|
| Ocean | -0.02350 | 0.11512 | OVER |
| Settlement | +0.02936 | 0.09198 | UNDER |
| Port | -0.00571 | 0.01558 | OVER |
| Ruin | +0.00197 | 0.01046 | UNDER |
| Forest | +0.00767 | 0.02238 | UNDER |
| Mountain | -0.00981 | 0.00981 | OVER |

### Feature-Residual Correlations (Pearson r, bold = |r| > 0.05)

| Feature | Ocean | Settlement | Port | Ruin | Forest | Mountain |
|---------|-------|-----------|------|------|--------|----------|
| dist_ocean | +0.01 | +0.00 | **-0.16** | **+0.09** | +0.01 | **+0.29** |
| dist_mountain | -0.03 | +0.02 | **+0.06** | +0.00 | -0.01 | **-0.07** |
| forest_cc_size | **+0.08** | +0.02 | -0.01 | +0.01 | **-0.14** | +0.03 |
| heterogeneity | +0.05 | **-0.07** | **+0.12** | **-0.05** | +0.00 | **-0.38** |
| n_forest | -0.02 | +0.05 | **-0.06** | +0.03 | -0.03 | **+0.14** |
| n_mountain | +0.03 | -0.04 | -0.03 | -0.00 | +0.02 | **+0.05** |
| dist_civ | -0.03 | +0.04 | **+0.06** | **-0.05** | -0.02 | **+0.40** |
| n_ocean | -0.02 | -0.04 | **+0.31** | -0.05 | -0.01 | **-0.59** |
| n_civ | **+0.05** | **-0.06** | -0.04 | -0.03 | +0.02 | **-0.08** |
