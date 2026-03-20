# Independent Analysis of Astar Island Cellular Automaton

**Data**: 40 ground truth files (8 rounds x 5 seeds), each 40x40 grid with 6-class probability tensor.
**Method**: Direct statistical analysis from raw data using numpy/scipy. No prior analysis consulted.

---

## 1. Static Cell Types (Never Change)

| Initial Type | GT Outcome | Confidence |
|---|---|---|
| **Ocean (10)** | Empty (class 0), P=1.000 | 8354/8354 cells deterministic |
| **Mountain (5)** | Mountain (class 5), P=1.000 | 1261/1261 cells deterministic |

These are perfectly deterministic across all 40 files with zero variance.

---

## 2. Probability Quantization

All ground truth probabilities are exact multiples of **0.005 = 1/200**. There are exactly 201 unique values (0.000, 0.005, ..., 1.000). This means the ground truth was generated from **200 Monte Carlo simulations** per grid.

---

## 3. Settlement Probability: Distance Decay

Settlement probability on non-static cells is dominated by **distance to nearest initial settlement** (Euclidean).

### 3a. Plains Cells (N=38,953 total)

| Distance | N | P(settlement) | P(ruin) | P(forest) | P(empty) |
|---|---|---|---|---|---|
| 1-2 | 9831 | 0.1865 | 0.0174 | 0.0547 | 0.7321 |
| 2-3 | 12336 | 0.1340 | 0.0141 | 0.0398 | 0.8028 |
| 3-4 | 7773 | 0.0945 | 0.0108 | 0.0290 | 0.8564 |
| 4-5 | 4234 | 0.0691 | 0.0082 | 0.0184 | 0.8951 |
| 5-6 | 2764 | 0.0503 | 0.0055 | 0.0108 | 0.9244 |
| 6-7 | 958 | 0.0378 | 0.0037 | 0.0071 | 0.9427 |
| 7-8 | 526 | 0.0278 | 0.0024 | 0.0044 | 0.9577 |
| 8-9 | 251 | 0.0242 | 0.0021 | 0.0024 | 0.9666 |
| 11+ | ~50 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

**Cutoff**: P(settlement) drops to exactly 0 at distances ~5-11 depending on round.

### 3b. Forest Cells (N=13,552 total)

Forest cells have ~5-18% **higher** P(settlement) than Plains at the same distance:

| Distance | Forest P(sett) | Plains P(sett) | Ratio |
|---|---|---|---|
| 1-2 | 0.1968 | 0.1865 | 1.055 |
| 3-4 | 0.0981 | 0.0945 | 1.039 |
| 5-6 | 0.0550 | 0.0503 | 1.094 |
| 7-8 | 0.0327 | 0.0278 | 1.178 |

Forest cells also retain high P(forest) that decreases near settlements (0.652 at d=1 to 1.000 at d=11+).

### 3c. Settlement Cells (Initial)

Initial settlement cells have the highest P(remaining settlement): mean 0.293, varying 0.018-0.422 by round.

### 3d. Functional Form of Distance Decay

Three models were compared (pooled across rounds for Plains cells):

| Model | Formula | SSE |
|---|---|---|
| **Inverse quadratic** | 0.2342 / (1 + 0.1172 * d^2) | **0.000225** |
| Exponential | 0.2867 * exp(-0.3025 * d) | 0.000501 |
| Power law | 0.2946 * d^(-1.006) | 0.000825 |

**Inverse quadratic is the best fit** pooled. However, the decay shape changes dramatically per round (see Section 6), so the functional form may differ per round.

---

## 4. Port Formation

### 4a. Ocean Adjacency is REQUIRED

Port probability is **exactly zero** for cells not adjacent to ocean (8-connected). Verified across all 31,779 non-ocean-adjacent Plains cells, all 11,245 non-ocean-adjacent Forest cells, and all 1,702 non-ocean-adjacent Settlement cells: max P(port) = 0.000.

### 4b. Number of Ocean Neighbors Matters

For ocean-adjacent cells within distance 5 of a settlement:

| N ocean neighbors | Port fraction | P(sett) | P(port) | P(sett)+P(port) | N |
|---|---|---|---|---|---|
| 1 | **0.074** | 0.1292 | 0.0103 | 0.1395 | 1327 |
| 2 | **0.523** | 0.0601 | 0.0659 | 0.1260 | 921 |
| 3 | **0.537** | 0.0668 | 0.0775 | 0.1443 | 4189 |
| 4 | **0.514** | 0.0575 | 0.0609 | 0.1183 | 411 |
| 5 | **0.478** | 0.0523 | 0.0480 | 0.1003 | 240 |

**Key finding**: There's a sharp threshold at N_ocean >= 2. With 1 ocean neighbor, only ~7% of the "settlement influence" produces ports. With 2+, it's ~50%.

### 4c. Port Distance Decay

P(port) follows the same distance decay as P(settlement), just redirected. The total P(settlement) + P(port) for ocean-adjacent cells roughly equals P(settlement) for non-ocean-adjacent cells at the same distance.

### 4d. Ports Act as Settlement Sources

Initial ports radiate settlement/port influence similar to settlements. Cells near ports but far from settlements show nonzero P(settlement) and P(port), confirming ports are equivalent sources.

---

## 5. Ruin Formation

### 5a. Where Ruins Appear

Ruins appear on ALL non-static cell types, with probability proportional to settlement influence:

| Initial Type | Mean P(ruin) | % cells with P(ruin) > 0.001 |
|---|---|---|
| Settlement | 0.0260 | 91.7% |
| Port | 0.0217 | 88.5% |
| Forest | 0.0130 | 68.4% |
| Plains | 0.0125 | 67.1% |

### 5b. Ruin Fraction

The ratio P(ruin) / (P(settlement) + P(ruin)) is roughly constant across distances:

- d=1-2: 0.096
- d=2-3: 0.101
- d=3-4: 0.105
- d=4-5: 0.103
- d=5-6: 0.093

**Mean ruin fraction ~0.10** (10% of settlement events produce ruins instead). This varies by round (0.07-0.16).

### 5c. Correlation with Settlement

Pearson correlation between P(ruin) and P(settlement) across all cells: **0.663**. P(ruin) tracks settlement influence but is not a simple fixed fraction of it.

---

## 6. Round-by-Round Variation (CRITICAL)

Rounds differ **dramatically** in settlement intensity. This is the most important feature for prediction.

### 6a. Overall Settlement Intensity by Round

| Round | Plains mean P(sett) | Sett survival P(sett) | Forest survival P(forest) | Ruin fraction | Forest fraction |
|---|---|---|---|---|---|
| 1 | 0.1558 | 0.4100 | 0.7449 | 0.077 | 0.043 |
| 2 | 0.1899 | 0.4098 | 0.6627 | 0.091 | 0.063 |
| **3** | **0.0021** | **0.0184** | **0.9711** | **0.163** | **0.010** |
| 4 | 0.0902 | 0.2340 | 0.8179 | 0.087 | 0.043 |
| 5 | 0.1205 | 0.3275 | 0.7686 | 0.102 | 0.045 |
| 6 | **0.2459** | **0.4125** | **0.5521** | **0.121** | **0.092** |
| 7 | 0.1380 | 0.4216 | 0.7535 | 0.074 | 0.036 |
| **8** | **0.0245** | **0.0686** | **0.9010** | **0.139** | **0.035** |

**Round 3** is extremely "harsh" -- almost no settlement activity (P(sett) for plains = 0.002).
**Round 6** is the most "expansive" -- highest settlement probability at all distances.
**Round 8** is also quite suppressed.

### 6b. The Decay Rate ALSO Changes Per Round

The ratio of P(sett) between distances is NOT constant across rounds. This means rounds differ in both **amplitude** and **decay rate**:

| Round | Exponential A | Exponential B (decay rate) | Effective reach (1/B) |
|---|---|---|---|
| 1 | 0.402 | 0.326 | 3.1 |
| 2 | 0.305 | 0.165 | 6.1 |
| 3 | 0.244 | 2.264 | 0.4 |
| 4 | 0.207 | 0.308 | 3.2 |
| 5 | 0.626 | 0.658 | 1.5 |
| 6 | 0.368 | 0.141 | 7.1 |
| 7 | 1.588 | 1.070 | 0.9 |
| 8 | 0.067 | 0.364 | 2.7 |

The correlation between A and B is only 0.19 -- they are nearly independent. **This means P(sett) = f(dist) * g(round) does NOT hold.** The whole distance-decay curve changes shape per round.

### 6c. Ratio Test (Non-Factorizability)

If P(sett) = f(dist) * g(round), then P(r6,d)/P(r1,d) should be constant for all d. It is NOT:

| Distance | R1 P(sett) | R6 P(sett) | Ratio R6/R1 |
|---|---|---|---|
| 1-2 | 0.2263 | 0.2950 | 1.30 |
| 3-4 | 0.1543 | 0.2273 | 1.47 |
| 5-6 | 0.0578 | 0.1597 | 2.76 |
| 7-8 | 0.0180 | 0.1322 | 7.36 |

### 6d. Round Effect is NOT Related to Grid Composition

Correlation between settlement density in the initial grid and mean P(settlement) = **0.376** (weak). The round effect is a hidden/latent parameter not observable from the initial grid.

---

## 7. Competition Between Settlements

### 7a. Settlement Density Suppresses Survival

For initial Settlement cells, controlling for round:

| N settlements within dist 5 | Mean P(remain settlement) |
|---|---|
| 0 | ~0.43 (isolated) |
| 1 | ~0.43 |
| 2 | ~0.39 |
| 3 | ~0.35 |
| 4 | ~0.30 |

Higher density = lower individual survival. Effect is ~25% reduction from 0 to 4 neighbors.

### 7b. Multiple Nearby Settlements Do NOT Boost P(sett) for Empty Cells

At fixed nearest distance, having more settlements nearby **slightly decreases** P(settlement):

- d=1-2, phi_sum=[0.6,0.8): P(sett)=0.192
- d=1-2, phi_sum=[1.4,1.6): P(sett)=0.175
- d=1-2, phi_sum=[1.8,2.0): P(sett)=0.134

This is consistent with a competition/resource-sharing model rather than additive influence.

---

## 8. Settlement Neighbor Count Effect

For Plains cells at distance 1-2 (all adjacent to at least 1 settlement):

| N settlement neighbors | P(settlement) | N |
|---|---|---|
| 1 | 0.1848 | 9359 |
| 2 | 0.2203 | 472 |

Having 2 settlement neighbors boosts P(settlement) by ~19% compared to 1 neighbor. However, the sample with 2 neighbors is small and this may partly reflect that such cells are in denser clusters.

---

## 9. Edge Effects

Cells near the grid border (within 1 cell of edge) have significantly lower P(settlement):

| Position | Distance 1-2 | Distance 2-3 | Distance 3-4 |
|---|---|---|---|
| Border (0-1) | 0.097 | 0.080 | 0.056 |
| Interior | 0.191 | 0.138 | 0.099 |

Border cells show ~50% reduction. This is likely because the border reduces the number of valid pathways/neighbors.

---

## 10. Forest Dynamics

### 10a. Forest Does NOT Spread Independently

P(forest) for Plains cells is virtually **independent** of the number of forest neighbors:

| N forest neighbors | P(forest) at d_sett=1-2 |
|---|---|
| 0 | 0.0552 |
| 1 | 0.0539 |
| 2 | 0.0556 |
| 3 | 0.0546 |
| 4 | 0.0541 |
| 5 | 0.0526 |

**Forest adjacency has NO effect on P(forest).** Forest probability is entirely driven by settlement proximity (inverse relationship).

### 10b. Forest is the "Default Reclamation"

The forest fraction of the "remainder" (P(forest) / (P(empty) + P(forest))) also depends on distance:

| Distance | Forest fraction |
|---|---|
| 1 | 0.076 |
| 3 | 0.038 |
| 5 | 0.012 |
| 7 | 0.003 |
| 8+ | ~0 |

At large distances (11+), far-from-settlement Plains cells become P(empty)=1.000 exactly, with no forest.

### 10c. Forest on Initial Forest Cells

Initial forest cells preserve forest at high rates that depend on settlement distance and round. At d=11+ from settlements, P(forest)=1.000 always.

---

## 11. Mountain Adjacency

Mountain adjacency has **minimal effect** on P(settlement). Controlling for distance to settlement:

| N mountain neighbors | P(sett) at d=1-2 | N |
|---|---|---|
| 0 | 0.1874 | 9169 |
| 1 | 0.1802 | 366 |
| 2 | 0.1758 | 190 |
| 3 | 0.1479 | 73 |

Small reduction (~5-20%) with many mountain neighbors, but sample sizes for 3+ are small.

---

## 12. Non-Obvious Patterns

1. **Settlements never spawn adjacent to each other in the initial grid**: All 1802 settlement cells have 0 settlement neighbors. Min inter-settlement distance is always >= 2.

2. **All seeds have different geographies**: No two (round, seed) pairs share the same initial grid.

3. **Grid sizes are consistent**: Always exactly 40x40. Typical composition: ~40-55 settlements, 0-5 ports, ~300-370 forest, ~930-1020 plains, ~15-50 mountains, ~180-250 ocean.

4. **Forest probability near settlements is NOT forest spread** -- it's a byproduct of the settlement formation process. The simulation appears to sometimes place forest instead of empty as part of the settlement dynamics.

5. **Ruin fraction increases in "harsh" rounds**: Round 3 (lowest settlement) has the highest ruin fraction (0.163), while high-settlement rounds like 7 have the lowest (0.074). This suggests ruins are "failed settlements" and harsh conditions produce proportionally more failures.

6. **Port fraction is remarkably stable**: The ~50% port fraction for cells with 2+ ocean neighbors holds across most rounds (except round 3 which has too little activity to measure).

---

## 13. Practical Implications for KL Divergence Prediction

### What we need to predict:
For a new grid, we must output a 40x40x6 probability tensor that minimizes KL divergence from the true (unknown) distribution.

### Key modeling recommendations:

1. **Ocean -> [1,0,0,0,0,0] and Mountain -> [0,0,0,0,0,1]**: These are deterministic. Get them exact.

2. **Round parameter is LATENT**: The round effect cannot be determined from the grid. We need to either:
   - Estimate it from the API response pattern
   - Use a prior distribution over round parameters
   - Average over possible round parameters (mixture model)

3. **The model structure for each non-static cell**:
   - Compute distance to nearest settlement/port (Euclidean)
   - Check ocean adjacency and count ocean neighbors
   - Check if initial cell is settlement, port, forest, or plains
   - Apply round-specific distance decay to get base P(settlement_event)
   - Split settlement_event into P(settlement), P(ruin), P(port) based on:
     - Ruin fraction: ~0.10 (round-dependent)
     - Port fraction: ~0.50 if ocean_adj with N_ocean >= 2, ~0.07 if N_ocean=1, 0 otherwise
   - Remaining probability splits between P(empty) and P(forest), with forest fraction dependent on distance and round

4. **Competition effect**: High-density settlement regions suppress individual cell probabilities. Consider implementing a local density correction.

5. **Edge effect**: Border cells need a ~50% reduction in settlement probability.

6. **Quantization**: Predictions should be output as multiples of 0.005 to match the ground truth format.
