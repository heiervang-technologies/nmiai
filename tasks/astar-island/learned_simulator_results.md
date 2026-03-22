# Learned Simulator Results

## Summary

Built a cellular automaton simulator that learns per-step transition rules from 90 replay files (18 rounds x 5 seeds x 51 frames each = 4,500 frame-to-frame transitions).

**Best CV wKL: 0.118** (vs target 0.032, vs current best UNet 0.032)

## Approach

### 1. Replay Data Analysis
- 90 replay files, each with 51 frames (step 0-50)
- Cell types: Settlement(1), Port(2), Ruin(3), Forest(4), Mountain(5), Ocean(10), Plains(11)
- Mountain and Ocean are immutable (never change)
- ~800 mutable cells per grid x 50 transitions x 90 replays = ~3.6M transition samples

### 2. Per-Step Transition Tables
Dense numpy array indexed by `(step, current_class, n_civ_neighbors_bin, n_forest_neighbors_bin, n_ocean_neighbors_bin, dist_to_civ_bin)` mapping to probability distribution over 6 output classes.

- Table shape: (50, 6, 5, 5, 4, 4, 6) = 120,000 entries
- Hierarchical fallback: fine -> coarse (drop forest) -> coarser (drop ocean) -> global (step + class only)
- Minimum sample thresholds: 5 (fine), 3 (coarse1), 2 (coarse2)

### 3. Vectorized Monte Carlo Simulation
- Fully vectorized: no per-cell Python loops during simulation
- Neighbor counting via padded array shifts
- Multinomial sampling via cumulative probability + uniform random
- 200 simulations in ~6 seconds per grid

### 4. GT-Replay Blend
- Ground truth lookup tables provide calibrated base predictions
- MC simulation adds dynamics-aware corrections
- Best blend: 30% MC + 70% GT lookup (wKL = 0.118)

## Cross-Validation Results (Leave-One-Round-Out)

| Round | KL     | wKL    | Notes |
|-------|--------|--------|-------|
| R1    | 0.0572 | 0.0614 | Moderate growth |
| R2    | 0.0617 | 0.0762 | Moderate growth |
| R3    | 0.3142 | 0.1186 | Harsh (mass death) |
| R4    | 0.0551 | 0.0424 | |
| R5    | 0.0747 | 0.0580 | |
| R6    | 0.1342 | 0.1961 | High growth + conflict |
| R7    | 0.1496 | 0.1485 | |
| R8    | 0.1456 | 0.0721 | Harsh |
| R9    | 0.0395 | 0.0432 | |
| R10   | 0.2523 | 0.1002 | Harsh |
| R11   | 0.1316 | 0.1753 | Very prosperous |
| R12   | 0.2446 | 0.2319 | High variance |
| R13   | 0.0538 | 0.0444 | |
| R14   | 0.1490 | 0.2096 | High growth + conflict |
| R15   | 0.0451 | 0.0534 | |
| R16   | 0.0975 | 0.0496 | |
| R17   | 0.0986 | 0.1355 | |
| R18   | 0.2604 | 0.3487 | Very prosperous (hardest) |
| R19   | 0.2076 | 0.0802 | Harsh |
| **Mean** | **0.1354** | **0.1182** | |

## Key Findings

### 1. Regime Detection is Impossible from Initial State
- Initial settlement populations, food, wealth, defense are nearly identical across regimes
- Same initial grid produces vastly different outcomes depending on sim_seed
- Correlation between initial features and growth rate: |r| < 0.3
- **Implication**: Cannot predict whether a round will be harsh/moderate/prosperous

### 2. Step-Dependent Dynamics
- Changes per step increase over time: ~5 changes at step 0, ~50 at step 49
- Settlements die early (steps 0-10), survivors grow later (steps 20-49)
- Phase structure exists but is stochastic, not deterministic

### 3. Error Analysis
- **Worst rounds**: R18 (wKL=0.349), R12 (0.232), R14 (0.210) - all high-growth rounds
- **Best rounds**: R9 (wKL=0.043), R4 (0.042), R13 (0.044) - moderate rounds
- High-growth rounds have more spatial variation that simple features can't capture
- Neural approaches (UNet) capture spatial patterns that feature engineering misses

### 4. Why 0.032 is Hard Without Neural Networks
- The UNet predictor achieves 0.032 by learning spatial convolution patterns
- These patterns capture settlement expansion direction, terrain interactions
- Simple neighborhood features miss long-range spatial correlations
- The 5 replay seeds per round provide only marginal statistical power

## Method Comparison

| Method | CV wKL | Time/pred |
|--------|--------|-----------|
| GT lookup only | 0.126 | 0.1s |
| Replay lookup only | 0.127 | 0.5s |
| MC simulation only | 0.119 | 6s |
| **MC + GT blend (0.3/0.7)** | **0.118** | **6s** |
| UNet (neural) | 0.032 | 0.1s |

## File

- Simulator: `tasks/astar-island/learned_simulator.py`
- Usage: `predict(initial_grid) -> np.ndarray (40, 40, 6)`
- Validation: `python3 learned_simulator.py --validate --blend 0.3 --n-sims 200`
