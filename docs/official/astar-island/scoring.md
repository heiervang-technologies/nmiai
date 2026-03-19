# Astar Island - Scoring

## Ground Truth

Ground truth distributions are computed from **hundreds of simulations** per seed, capturing the stochastic variance in outcomes.

## KL Divergence

Predictions are evaluated using **KL divergence** from the ground truth distribution:

```
KL(p || q) = sum(p_i * log(p_i / q_i))
```

Where:
- `p` is the ground truth distribution
- `q` is your predicted distribution

## Entropy Weighting

KL divergence is **entropy-weighted**: cells with higher entropy (more uncertain ground truth) contribute proportionally more to the score. This rewards getting the hard-to-predict cells right.

## Static Cell Exclusion

**Static cells** (cells that never change across simulations, e.g., ocean borders) are **excluded** from scoring. Only dynamic cells contribute to the final score.

## Score Formula

```
score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
```

Where `weighted_kl` is the entropy-weighted average KL divergence across all scored cells.

## Round Score

A round score is the **average score across all 5 seeds**.

## Leaderboard

The leaderboard displays each team's **best round score**.

## Zero Probability Warning

A predicted probability of **0.0** for any class that has non-zero ground truth probability results in **infinite KL divergence**, which gives a score of **zero** for that cell.

Always use a minimum probability floor of **0.01** for all classes in all cells.
