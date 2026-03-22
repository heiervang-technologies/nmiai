# Surrogate Monte Carlo with Spatial Correlations - Results

## Approach

The predictor captures **spatial correlations** in settlement growth that independent bucket priors miss.

### Key Innovation
Standard bucket priors compute `P(final_class | initial_type, distance, coast, regime)` independently per cell. But settlement growth is **clustered** - if one cell becomes a settlement, its neighbor is more likely to also become one.

This predictor adds a **conditional transition table**:
`P(final_class | initial_type, distance, coast, n_civ_neighbors_in_final_state)`

### Method
1. **Base tables**: Hierarchical bucket priors (type, distance, ocean count, coast) with regime conditioning
2. **Spatial tables**: Same features + quantized neighbor civ count from GT argmax
3. **Gibbs sampling**:
   - Sample initial grid from base prior
   - For each sweep: compute neighbor civ counts, blend spatial conditional with base, resample
   - 2 sweeps per simulation, 50-200 simulations
4. **Blending**: Final prediction = 35% MC + 65% base prior (reduces sampling noise)

## Leave-One-Round-Out CV Results (50 sims, 2 sweeps, blend=0.65)

| Round | Spatial wKL | Base wKL | Delta | Winner |
|-------|------------|----------|-------|--------|
| R01 | 0.0520 | 0.0523 | -0.0003 | Spatial |
| R02 | 0.0707 | 0.0799 | -0.0091 | Spatial |
| R03 | 0.1329 | 0.1159 | +0.0170 | Base |
| R04 | 0.0450 | 0.0319 | +0.0131 | Base |
| R05 | 0.0476 | 0.0436 | +0.0040 | Base |
| R06 | 0.1726 | 0.1894 | -0.0169 | Spatial |
| R07 | 0.1218 | 0.1344 | -0.0126 | Spatial |
| R08 | 0.0750 | 0.0608 | +0.0141 | Base |
| R09 | 0.0426 | 0.0377 | +0.0049 | Base |
| R10 | 0.1127 | 0.0955 | +0.0172 | Base |
| R11 | 0.1422 | 0.1708 | -0.0287 | Spatial |
| R12 | 0.1914 | 0.2197 | -0.0284 | Spatial |
| R13 | 0.0468 | 0.0334 | +0.0134 | Base |
| R14 | 0.1669 | 0.1871 | -0.0202 | Spatial |
| R15 | 0.0501 | 0.0558 | -0.0057 | Spatial |
| R16 | 0.0458 | 0.0379 | +0.0078 | Base |
| R17 | 0.1116 | 0.1341 | -0.0225 | Spatial |
| R18 | 0.2901 | 0.3330 | -0.0429 | Spatial |

### Summary
- **Overall spatial wKL: 0.1065** (score 89.35)
- **Overall base wKL: 0.1119** (score 88.81)
- **Delta: -0.0053** (spatial is better)
- **Win rate: 10/18 rounds** (56%)
- Spatial wins are larger magnitude (avg -0.019) vs losses (avg +0.010)

### Observations
- Spatial MC helps most on **high-entropy rounds** (prosperous regimes with lots of frontier expansion): R11, R12, R14, R17, R18
- Spatial MC hurts on **low-entropy rounds** where base prior is already very confident
- The 65% base blend is critical - pure MC (blend=0) has too much sampling noise
- 2 Gibbs sweeps is optimal; more sweeps can cause over-coupling

## Configuration
- `N_SIMS = 200` (50 for fast CV, 200 for production)
- `N_SWEEPS = 2`
- `MC_BLEND = 0.65` (65% base + 35% MC)
- `SPATIAL_WEIGHT = 0.5` (blend of spatial conditional vs base in Gibbs step)

## API
```python
from surrogate_mc import predict
prob_tensor = predict(initial_grid)  # (40, 40, 6) ndarray
```
