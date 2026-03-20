# Astar Island Automaton: Exact Behavior Patterns

Analysis of 40 ground truth maps (8 rounds x 5 seeds), 64,000 total cells.

## 0. CRITICAL DISCOVERY: Variable Round Regimes

**The simulation parameters change between rounds.** Settlement survival rates vary 23x between rounds, from 1.8% to 42.2%. Rounds fall into three regimes:

| Regime | Rounds | Settlement Survival | Growth Rate (Plains) |
|--------|--------|--------------------|-----------------------|
| Prosperous | 1, 2, 6, 7 | 41.4% | 15.6-24.6% at dist=1 |
| Moderate | 4, 5 | 23.4-32.8% | 9.0-12.1% at dist=1 |
| Harsh | 3, 8 | 1.8-6.9% | 0.2-2.5% at dist=1 |

Variation is consistent across all 5 seeds within a round (e.g., Round 3 seeds: 1.4%, 1.7%, 1.8%, 1.9%, 2.3% survival). Map characteristics (settlement count, forest, ocean) are similar across rounds -- the difference is simulation parameters, not map structure.

**Implication for prediction:** Without knowing the current round's regime, you must use a mixture or infer it from observations.

## 1. STATIC TERRAIN (100% deterministic)

- **Mountains**: Always remain Mountain. P(mountain)=1.0000 across all 1,261 mountain cells.
- **Ocean**: Always becomes Empty (class 0). P(empty)=1.0000 across all 8,354 ocean cells.
- These provide zero bits of uncertainty -- exclude from scoring.

## 2. SETTLEMENT GROWTH RULES

### 2.1 Distance Decay Function

Settlement probability follows an exponential decay from initial settlement positions:

```
P(settlement | plains, distance d) = A * exp(-B * d)
```

| Regime | A (amplitude) | B (decay rate) | Half-life distance |
|--------|--------------|----------------|-------------------|
| Prosperous | 0.3631 | 0.3060 | 2.3 cells |
| Moderate | 0.3008 | 0.5131 | 1.4 cells |
| Harsh | 0.0361 | 0.4451 | 1.6 cells |

### 2.2 Plains Cell: P(Settlement) by Distance (All Rounds Averaged)

| Distance | P(settlement) | P(empty) | P(ruin) | P(forest) | P(port) | n |
|----------|--------------|----------|---------|-----------|---------|---|
| 1 | 0.1852 | 0.7327 | 0.0173 | 0.0545 | 0.0103 | 10,100 |
| 2 | 0.1333 | 0.8033 | 0.0140 | 0.0396 | 0.0098 | 12,646 |
| 3 | 0.0937 | 0.8580 | 0.0107 | 0.0287 | 0.0090 | 7,825 |
| 4 | 0.0674 | 0.8990 | 0.0080 | 0.0173 | 0.0082 | 4,095 |
| 5 | 0.0452 | 0.9334 | 0.0052 | 0.0090 | 0.0072 | 2,537 |
| 6 | 0.0299 | 0.9570 | 0.0029 | 0.0044 | 0.0058 | 842 |
| 7 | 0.0167 | 0.9783 | 0.0012 | 0.0015 | 0.0023 | 459 |
| 8 | 0.0100 | 0.9865 | 0.0008 | 0.0007 | 0.0020 | 220 |
| 9 | 0.0075 | 0.9910 | 0.0006 | 0.0002 | 0.0008 | 108 |
| 10+ | 0.0033 | 0.9957 | 0.0003 | 0.0002 | 0.0004 | 63 |
| 12+ | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 34 |

**At distance >= 12 from any initial settlement, plains cells are deterministically Empty.**

### 2.3 Forest Cell: P(Settlement) by Distance

| Distance | P(settlement) | P(forest) | P(empty) | P(ruin) | n |
|----------|--------------|-----------|----------|---------|---|
| 1 | 0.1963 | 0.6518 | 0.1224 | 0.0181 | 3,585 |
| 2 | 0.1367 | 0.7534 | 0.0865 | 0.0146 | 4,513 |
| 3 | 0.0968 | 0.8189 | 0.0649 | 0.0111 | 2,640 |
| 4 | 0.0710 | 0.8764 | 0.0378 | 0.0078 | 1,353 |
| 5 | 0.0492 | 0.9180 | 0.0206 | 0.0052 | 863 |
| 6 | 0.0304 | 0.9532 | 0.0087 | 0.0030 | 292 |
| 7 | 0.0197 | 0.9730 | 0.0041 | 0.0016 | 155 |
| 8+ | 0.0082 | 0.9875 | 0.0018 | 0.0007 | 75 |
| 12+ | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 11 |

**Key finding:** Forest cells have SIMILAR settlement growth probability as plains at the same distance, but retain forest probability (unlike plains which become empty). Forest decay: P(settle) = 0.2834 * exp(-0.3476 * d).

### 2.4 Source Type

- 73.5% of new settlements come from Plains cells
- 26.5% come from Forest cells
- 0% from Mountain, Ocean, or Empty

### 2.5 Neighbor Count Effect

Initial settlements are ALWAYS isolated from each other (0 out of 1,802 have adjacent settlement neighbors). Growth is driven by distance, not discrete neighbor counting.

### 2.6 Density / Competition Effect

**Counterintuitive: More nearby settlements REDUCE growth probability.**

At dist=1 from nearest settlement (plains, prosperous rounds), controlling for nearest distance:
- Low total influence (few settlements nearby): P(settle)=0.2914
- High total influence (many settlements nearby): P(settle)=0.2398

At dist=2:
- Low total influence: P(settle)=0.2287
- High total influence: P(settle)=0.1766

**Interpretation:** Settlements compete for territory. More competing settlements in an area means each one's growth is suppressed. This is consistent with the "conflict/raids" phase in the simulation.

### 2.7 Directional Symmetry

Settlement growth is isotropic (no directional bias):
- North: P(settle)=0.1958
- South: P(settle)=0.1955
- East: P(settle)=0.1974
- West: P(settle)=0.1955

Cardinal vs diagonal: Cardinals slightly favored (0.1960 vs 0.1800) but difference is small.

## 3. PORT RULES

### 3.1 Ocean Adjacency: STRICTLY REQUIRED

**Zero cells without ocean adjacency have P(port) > 0.** This is the hardest rule in the system. Port formation requires at least one ocean cell in the Moore neighborhood (8-connected).

### 3.2 Port Probability by Distance to Settlement

Ports form near settlements on coastlines:

**Prosperous rounds:**
| Distance to settlement | P(port) | n |
|----------------------|---------|---|
| 0 (settlement itself) | 0.1656 | 88 |
| 1 | 0.1475 | 729 |
| 2 | 0.1006 | 1,183 |
| 3 | 0.0735 | 986 |
| 4 | 0.0574 | 670 |
| 5 | 0.0415 | 532 |
| 7+ | 0.0103 | 124 |

Decay fit: P(port) = 0.0192 * exp(-0.1225 * d) -- much slower decay than settlement.

### 3.3 Port Count per Map

Mean: 1.3 ports per map (argmax). Range: 0-7. Most maps (60%) have 0 ports by argmax. In harsh rounds, essentially zero ports form.

### 3.4 Coastal Settlement Penalty

Settlements adjacent to ocean have REDUCED survival: P(settle) drops from 0.297 (0 ocean neighbors) to 0.064 (3+ ocean neighbors). The coast converts settlement probability into port probability.

## 4. RUIN RULES

### 4.1 Ruin Probability by Distance to Initial Settlement

Ruins are a minor outcome, peaking at the settlement itself:

| Distance | P(ruin) | n |
|----------|---------|---|
| 0 (settlement) | 0.0258 | 1,880 |
| 1 | 0.0175 | 13,685 |
| 2 | 0.0142 | 17,159 |
| 3 | 0.0108 | 10,465 |
| 4 | 0.0080 | 5,448 |
| 5 | 0.0052 | 3,400 |
| 7+ | 0.0013 | 614 |
| 10+ | ~0 | varies |

Decay fit: P(ruin) = 0.0255 * exp(-0.3262 * d) for all-rounds plains.

### 4.2 Ruin Origin

Ruins appear on cells that initially held ANY type:
- Initial settlements: 0.0334 mean P(ruin)
- Initial forests: 0.0273
- Initial plains: 0.0266

Ruins represent collapsed/raided settlements -- they appear where settlements briefly existed before being destroyed.

### 4.3 Ruin by Regime

| Regime | P(ruin) at settlements | P(ruin) at plains dist=1 |
|--------|----------------------|--------------------------|
| Prosperous | 0.0355 | 0.0235 |
| Moderate | 0.0240 | 0.0173 |
| Harsh | 0.0073 | 0.0040 |

**In harsh rounds, fewer ruins because settlements barely grow at all -- nothing to ruin.**

## 5. FOREST DYNAMICS

### 5.1 Forest Persistence

Forest is highly persistent. At dist >= 5 from any settlement, P(forest) > 0.92. At dist >= 8, P(forest) > 0.99. Distant forests are essentially static.

### 5.2 Forest Clearing

Near settlements, forests are cleared to make room for settlement growth:
- dist=1: P(forest remains) = 0.6518
- dist=2: P(forest remains) = 0.7534
- dist=3: P(forest remains) = 0.8189

### 5.3 Forest Reclamation

Dead settlements become forest at significant rates:
- Prosperous rounds: 16.9% of initial settlements become forest
- Harsh rounds: 30.1% of initial settlements become forest

Plains cells near settlements also gain some forest probability, but it decays quickly:
- dist=1: P(forest)=0.054 (prosperous), 0.044 (harsh)
- dist=3: P(forest)=0.038 (prosperous), 0.017 (harsh)

### 5.4 Forest Does NOT Spread

Forest probability for plains cells is essentially independent of distance to initial forest:
- dist_to_forest=1: P(forest)=0.035
- dist_to_forest=2: P(forest)=0.035
- dist_to_forest=3: P(forest)=0.029

Forest probability on plains is driven by settlement dynamics (reclamation of cleared land), not by forest spreading from existing forests.

## 6. SETTLEMENT SURVIVAL (Initial Settlements)

### 6.1 By Regime

| Regime | P(survive as settlement) | P(become empty) | P(become forest) | P(become ruin) |
|--------|-------------------------|-----------------|-------------------|----------------|
| Prosperous | 0.4138 | 0.3763 | 0.1694 | 0.0355 |
| Moderate | 0.2814 | 0.4613 | 0.2293 | 0.0240 |
| Harsh | 0.0441 | 0.6470 | 0.3010 | 0.0073 |

### 6.2 Survival Factors

**Weak factors (small effects):**
- More forest neighbors slightly increase survival (0.263 with 0 forest vs 0.353 with 6 forest neighbors)
- More mountain neighbors slightly decrease survival (0.297 with 0 vs 0.232 with 3)
- More available land (non-ocean, non-mountain) increases survival (0.157 with 4 land vs 0.303 with 8 land)

**Strong factor: Competition**
- 0 settlements within radius 3: P(survive)=0.311
- 1 settlement within radius 3: P(survive)=0.282
- 2 settlements within radius 3: P(survive)=0.248
- 3 settlements within radius 3: P(survive)=0.191

**Coastal penalty:**
- 0 ocean neighbors: P(survive)=0.297
- 1 ocean neighbor: P(survive)=0.304
- 2 ocean neighbors: P(survive)=0.117 (but gains P(port)=0.152)
- 3+ ocean neighbors: P(survive)=0.064 (but gains P(port)=0.155)

### 6.3 Distance to Other Settlements

Surprisingly, distance to nearest other settlement has minimal effect on survival (0.275-0.325 range across dist 2-8). The competition effect is more about LOCAL density than pairwise distance.

## 7. OCEAN PROXIMITY EFFECT

Controlling for distance to nearest settlement, being ocean-adjacent REDUCES P(settlement) and ADDS P(port):

Plains at dist=1 to nearest settlement:
- dist_ocean=1: P(settle)=0.109, P(port)=0.077
- dist_ocean=2: P(settle)=0.216, P(port)=0.000
- dist_ocean=3+: P(settle)=0.180-0.190, P(port)=0.000

**Port probability is ZERO for non-ocean-adjacent cells.** At ocean distance >= 2, port probability is exactly 0.

## 8. HIDDEN PATTERNS ANALYSIS

### 8.1 No Diagonal Symmetry
Growth is radially symmetric (no N/S/E/W bias). Cardinal directions slightly favored over diagonals (0.196 vs 0.180).

### 8.2 Distance-Based Decay Functions
All dynamic probabilities follow exponential decay from initial settlement positions. No evidence of power-law or polynomial decay.

### 8.3 Competition is Local and Subtractive
More settlements nearby = less growth per settlement. This is NOT additive -- settlements compete rather than cooperate. Correlation of P(settle) with sum(1/dist) across all settlements is only r=0.20, vs r=0.51 for 1/(nearest_dist+1).

### 8.4 No Evidence of Non-Local Interactions
No trade routes, river paths, or long-range connections detected. All effects are local, decaying with Euclidean distance.

### 8.5 Phase Ordering
The simulation runs Growth -> Conflict -> Trade -> Winter -> Environment for 50 years. The net effect after 50 ticks is well-captured by the distance-based decay model.

## 9. PRACTICAL PREDICTION FORMULAS

### For an unknown regime (prior over regimes):

Given a plains cell at Euclidean distance `d` from nearest initial settlement and distance `d_ocean` from nearest ocean:

```
P(settlement) = w_pro * 0.363 * exp(-0.306*d) + w_mod * 0.301 * exp(-0.513*d) + w_harsh * 0.036 * exp(-0.445*d)
P(port) = (d_ocean <= 1) * [w_pro * 0.019 * exp(-0.123*d) + w_mod * 0.016 * exp(-0.361*d)]
P(ruin) = 0.026 * exp(-0.326*d) * regime_scale
P(forest) = (is_forest_initially) * [1 - P(settle) - P(port) - P(ruin) - P(empty)] + (is_plains) * 0.087 * exp(-0.418*d)
P(empty) = 1 - P(settlement) - P(port) - P(ruin) - P(forest) - P(mountain)
```

Where regime weights: w_pro ~ 0.5, w_mod ~ 0.25, w_harsh ~ 0.25 (from 4/8, 2/8, 2/8 rounds).

### For initial settlement cells:

Use the settlement survival table (Section 6.1) with regime mixture.

### For static cells:

- Mountain: [0, 0, 0, 0, 0, 1]
- Ocean: [1, 0, 0, 0, 0, 0]
- Plains at dist >= 12: [1, 0, 0, 0, 0, 0]
- Forest at dist >= 12: [0, 0, 0, 0, 1, 0]

## 10. FEATURE IMPORTANCE RANKING

Pearson correlation with P(settlement) on Plains cells:

| Feature | Correlation |
|---------|------------|
| 1/(dist_settle+1) | r = 0.4079 |
| dist_settle | r = -0.3951 |
| dist_ocean | r = 0.1175 |
| round (regime proxy) | r = -0.0956 |
| dist_mountain | r = -0.0437 |
| dist_forest | r = -0.0431 |

**Distance to nearest settlement explains ~16% of variance. Round regime is the second most important factor but cannot be predicted from the map.**

## 11. ENTROPY BY CELL TYPE

| Initial Type | Mean Entropy (bits) | Interpretation |
|-------------|--------------------|-|
| Settlement | 1.50 | Highly uncertain (most important to predict) |
| Port | 1.69 | Most uncertain (rare type) |
| Forest | 0.87 | Moderately uncertain near settlements |
| Plains | 0.73 | Mostly empty, some uncertainty near settlements |
| Mountain | 0.00 | Deterministic |
| Ocean | 0.00 | Deterministic |

**Scoring-wise, initial settlement cells and their immediate neighbors carry the most weight due to high entropy.**
