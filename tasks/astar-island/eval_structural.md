# Eval: Structural Zeros + Bayesian Regime Posterior

## Clean Eval Pipeline Results

New `eval.py` with proper leave-one-round-out CV. No file-hiding hacks.
Reproducible observations (fixed RNG seed=42).

### Summary Table

| Config | CV wKL | vs baseline |
|--------|--------|-------------|
| No observations | 0.1185 | baseline |
| 5 observations | 0.0780 | **-34%** |
| 5 obs + tau=20 | **0.0750** | **-37%** |

### Per-Round (5 obs + tau=20, our best config)

| Round | wKL | Regime | Quality |
|-------|-----|--------|---------|
| R10 | 0.018 | harsh | excellent |
| R3 | 0.021 | harsh | excellent |
| R4 | 0.036 | moderate | good |
| R13 | 0.039 | moderate | good |
| R8 | 0.043 | harsh | good |
| R5 | 0.053 | moderate | good |
| R9 | 0.058 | moderate | decent |
| R11 | 0.061 | prosperous | decent |
| R1 | 0.071 | moderate | decent |
| R14 | 0.081 | prosperous | ok |
| R2 | 0.092 | prosperous | ok |
| R6 | 0.109 | prosperous | weak |
| R7 | 0.145 | moderate | bad |
| R12 | 0.222 | moderate | terrible |

### R14 Specifically (Prosperous)

- No obs: wKL = 0.225
- 5 obs: wKL = 0.082 (64% improvement)
- 5 obs + tau=20: wKL = 0.081

Regime detection works well on R14 — correctly identifies prosperous regime.

### In-Sample vs CV Gap

| Round | In-sample | CV | Gap |
|-------|-----------|-----|-----|
| Mean | 0.1143 | 0.1185 | +0.004 (3.6%) |

Very small gap — model is not overfitting. The regime priors generalize well.

### Observations: Help 12/14 rounds, Hurt 2/14

| Helped most (obs vs no-obs) | |
|---|---|
| R3: 0.107 → 0.021 (-80%) | harsh, easy to detect |
| R10: 0.085 → 0.018 (-79%) | harsh, easy to detect |
| R11: 0.212 → 0.061 (-71%) | prosperous, clear signal |
| R14: 0.225 → 0.081 (-64%) | prosperous, clear signal |
| R6: 0.233 → 0.109 (-53%) | prosperous |

| Hurt slightly | |
|---|---|
| R1: 0.067 → 0.071 (+6%) | moderate, misclassified slightly |
| R9: 0.051 → 0.058 (+14%) | moderate, some noise |

### Previous eval issues (now fixed)

1. **Argmax observations were catastrophic** — settlement is rarely the argmax class even in prosperous rounds, so deterministic argmax made everything look harsh. wKL=0.293 (2.5x worse).
2. **File-hiding for CV was fragile** — race conditions with other agents. Now uses `build_model_from_data()` for clean in-memory CV.
3. **Old benchmark.py floor=0.01 masked structural zeros** — new eval uses floor=0.005.

### Recommended competition config

**regime_predictor + 5 observations + tau=20 → expected wKL ≈ 0.075**
