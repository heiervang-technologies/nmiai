# Astar Island Task - Overview

## Challenge Summary

Predict the future state of a simulated Viking island. Observe the island through limited viewport queries, then submit probability distributions over terrain types for each cell.

## Grid

- **40x40 grid** map
- **8 terrain types** that map to **6 prediction classes**

## Queries

- **50 queries per round**, shared across **5 seeds**
- Maximum viewport size: **15x15**

## Simulation

Each seed runs a **50-year simulation** of Viking settlement, conflict, trade, and environmental change.

## Submission

Submit a probability tensor of shape `[height][width][6]` representing the predicted probability distribution over the 6 classes for each cell.

```json
{
  "prediction": [
    [
      [0.1, 0.2, 0.3, 0.1, 0.2, 0.1],
      ...
    ],
    ...
  ]
}
```

## Important

**Never use 0.0 probability** for any class in any cell. A zero probability for a class that appears in the ground truth results in infinite KL divergence and a score of zero. Use a minimum floor of 0.01.
