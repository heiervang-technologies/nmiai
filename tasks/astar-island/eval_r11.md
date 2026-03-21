# Eval Boss R11 Report — Honest Out-of-Sample Results

## TRUE Out-of-Sample wKL (Leave-One-Round-Out CV)

Templates built from N-1 rounds, tested on held-out round. No fingerprinting.

### Template Predictor (smooth interpolation)

| Metric | No observations | 5 observations |
|--------|----------------|----------------|
| **Mean wKL** | **0.1000** | **0.0819** |

### Per-Round Breakdown

| Held-out | No obs wKL | 5 obs wKL | Delta | Verdict |
|----------|-----------|-----------|-------|---------|
| R1 | 0.0846 | 0.0964 | +0.012 | WORSE |
| R2 | 0.1082 | 0.1629 | +0.055 | WORSE |
| R3 | 0.0879 | 0.0245 | -0.063 | BETTER |
| R4 | 0.0335 | 0.0616 | +0.028 | WORSE |
| R5 | 0.0716 | 0.0820 | +0.010 | WORSE |
| R6 | 0.2538 | 0.0973 | -0.156 | BETTER |
| R7 | 0.1862 | 0.1106 | -0.076 | BETTER |
| R8 | 0.0490 | 0.0263 | -0.023 | BETTER |
| R9 | 0.0561 | 0.1116 | +0.056 | WORSE |
| R10 | 0.0694 | 0.0452 | -0.024 | BETTER |

### Key Finding

- Observations help on average (18% improvement) but HURT on 5/10 rounds
- Regime detection is a high-variance bet: when it picks the right template, huge gain; when wrong, significant loss
- Observations help most on "extreme" rounds (R3, R6, R7) where the pooled prior is far off
- Observations hurt on "moderate" rounds (R1, R2, R4, R5, R9) where the pooled prior is already decent

### R11 Recommendation

**Conservative (lower variance):** No observations. Expected wKL ~0.100.

**Aggressive (higher EV but risky):** 5 observations for regime detection. Expected wKL ~0.082, but could be anywhere from 0.025 to 0.163.

### Earlier Claims Corrected

The earlier reported "47% improvement from observations" was WRONG. That test used templates trained on ALL rounds including the test round. The observations were just helping select the already-memorized template. True out-of-sample improvement is 18% on average with high variance.
