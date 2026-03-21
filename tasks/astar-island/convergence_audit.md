# Astar Island Convergence Audit

Generated with np.random.seed(42), 15 rounds, 75 total seeds.

## Part 1: Overfitting / Collapse Audit

### 1.1 In-sample vs Leave-One-Round-Out CV

| Round | In-sample wKL | CV wKL | Gap % | Flag |
|-------|--------------|--------|-------|------|
| R1 | 0.067674 | 0.069547 | +2.8% | OK |
| R2 | 0.094988 | 0.102240 | +7.6% | OK |
| R3 | 0.110733 | 0.112048 | +1.2% | OK |
| R4 | 0.040469 | 0.042222 | +4.3% | OK |
| R5 | 0.061081 | 0.061480 | +0.7% | OK |
| R6 | 0.219832 | 0.226442 | +3.0% | OK |
| R7 | 0.154654 | 0.159422 | +3.1% | OK |
| R8 | 0.062267 | 0.061579 | -1.1% | OK |
| R9 | 0.054356 | 0.056816 | +4.5% | OK |
| R10 | 0.091056 | 0.091075 | +0.0% | OK |
| R11 | 0.201396 | 0.208849 | +3.7% | OK |
| R12 | 0.234913 | 0.245185 | +4.4% | OK |
| R13 | 0.044065 | 0.045721 | +3.8% | OK |
| R14 | 0.221657 | 0.223691 | +0.9% | OK |
| R15 | 0.073027 | 0.078234 | +7.1% | OK |
| **Overall** | **0.115478** | **0.118970** | **+3.0%** | |

### 1.2 Entropy Collapse Check

Fraction of dynamic cells where predicted entropy < 50% of GT entropy:

| Round | Collapse Fraction | Flag |
|-------|-------------------|------|
| R1 | 0.0% | OK |
| R2 | 0.8% | OK |
| R3 | 0.0% | OK |
| R4 | 0.0% | OK |
| R5 | 0.0% | OK |
| R6 | 3.0% | OK |
| R7 | 0.0% | OK |
| R8 | 0.0% | OK |
| R9 | 0.1% | OK |
| R10 | 0.0% | OK |
| R11 | 0.5% | OK |
| R12 | 0.0% | OK |
| R13 | 0.0% | OK |
| R14 | 0.0% | OK |
| R15 | 0.1% | OK |
| **Overall** | **0.3%** | |

### 1.3 Seed Variance Within Rounds

| Round | Mean wKL | Std | CV (std/mean) | Range | Flag |
|-------|----------|-----|---------------|-------|------|
| R1 | 0.067674 | 0.003612 | 0.053 | [0.0641, 0.0745] | OK |
| R2 | 0.094988 | 0.008505 | 0.090 | [0.0828, 0.1062] | OK |
| R3 | 0.110733 | 0.001997 | 0.018 | [0.1080, 0.1128] | OK |
| R4 | 0.040469 | 0.002701 | 0.067 | [0.0372, 0.0433] | OK |
| R5 | 0.061081 | 0.003706 | 0.061 | [0.0555, 0.0659] | OK |
| R6 | 0.219832 | 0.004635 | 0.021 | [0.2163, 0.2288] | OK |
| R7 | 0.154654 | 0.011597 | 0.075 | [0.1412, 0.1732] | OK |
| R8 | 0.062267 | 0.009335 | 0.150 | [0.0526, 0.0768] | OK |
| R9 | 0.054356 | 0.005492 | 0.101 | [0.0497, 0.0645] | OK |
| R10 | 0.091056 | 0.002907 | 0.032 | [0.0855, 0.0933] | OK |
| R11 | 0.201396 | 0.053071 | 0.264 | [0.1594, 0.3046] | OK |
| R12 | 0.234913 | 0.005450 | 0.023 | [0.2287, 0.2445] | OK |
| R13 | 0.044065 | 0.003822 | 0.087 | [0.0387, 0.0490] | OK |
| R14 | 0.221657 | 0.014400 | 0.065 | [0.1959, 0.2393] | OK |
| R15 | 0.073027 | 0.009839 | 0.135 | [0.0588, 0.0854] | OK |

### 1.4 Simulation vs Live Score Gap

- Simulated in-sample wKL: 0.115478
- Simulated CV wKL: 0.118970
- Live scores: R13=87.9, R15=88.6
- If score = 100*(1 - wKL), sim predicts score ~88.5
- Implied live wKL from scores: ~0.12
- Gap ratio (live/sim_CV): ~1.01x
- **Conclusion**: Simulation is optimistic -- live performance is worse than simulated, likely due to regime misclassification without ground truth observations.

## Part 2: Cheap Ensemble

Leave-one-round-out CV for each ensemble configuration:

| Config | Weights | Avg wKL | Worst-case wKL | Worst Round |
|--------|---------|---------|----------------|-------------|
| a) 100% regime | regime=1.0 | 0.118970 | 0.245185 | R12 |
| b) 80/20 regime+param | regime=0.8, parametric=0.2 | 0.116949 | 0.241650 | R12 |
| c) 70/15/15 regime+param+nb | regime=0.7, parametric=0.15, neighborhood=0.15 | 0.115764 | 0.239005 | R12 |
| d) 50/25/25 no-spatial | regime=0.5, parametric=0.25, neighborhood=0.25 | 0.113975 | 0.235432 | R12 |

### Recommendation

**Best by worst-case wKL**: d) 50/25/25 no-spatial
- Worst-case wKL: 0.235432 (R12)
- Average wKL: 0.113975

**Best by average wKL**: d) 50/25/25 no-spatial
- Average wKL: 0.113975
- Worst-case wKL: 0.235432 (R12)

### Per-round CV wKL by ensemble config

| Round | a) | b) | c) | d) |
|-------|-------|-------|-------|-------|
| R1 | 0.069547 | 0.068392 | 0.068106 | 0.067490 |
| R2 | 0.102240 | 0.099063 | 0.096555 | 0.093307 |
| R3 | 0.112048 | 0.113090 | 0.112627 | 0.113080 |
| R4 | 0.042222 | 0.042553 | 0.042159 | 0.042278 |
| R5 | 0.061480 | 0.060296 | 0.060147 | 0.059527 |
| R6 | 0.226442 | 0.219879 | 0.217054 | 0.211481 |
| R7 | 0.159422 | 0.156430 | 0.154963 | 0.152458 |
| R8 | 0.061579 | 0.062584 | 0.062708 | 0.063539 |
| R9 | 0.056816 | 0.055251 | 0.054154 | 0.052689 |
| R10 | 0.091075 | 0.092217 | 0.092058 | 0.092783 |
| R11 | 0.208849 | 0.202989 | 0.200291 | 0.195174 |
| R12 | 0.245185 | 0.241650 | 0.239005 | 0.235432 |
| R13 | 0.045721 | 0.045622 | 0.045067 | 0.044811 |
| R14 | 0.223691 | 0.218856 | 0.218162 | 0.214944 |
| R15 | 0.078234 | 0.075368 | 0.073405 | 0.070624 |
