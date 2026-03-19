# Tripletex Task - Overview

## Challenge Summary

Automate accounting tasks using the Tripletex API. Your solution receives a natural language prompt describing an accounting task and must execute it correctly against a Tripletex sandbox.

## Task Structure

- **30 task types** covering various accounting operations
- **56 variants** of each task type (7 languages x 8 datasets)
- **5-minute timeout** per submission
- **Score range**: 0 - 6.0 per task

A fresh sandbox environment is provisioned for each submission.

## Scoring Overview

Score is calculated as:

```
score = correctness * tier_multiplier + efficiency_bonus
```

- **Correctness**: Normalized 0-1 based on field-by-field verification
- **Tier multiplier**: Increases over time:
  - **Tier 1** (x1) - Available now
  - **Tier 2** (x2) - Unlocked early Friday
  - **Tier 3** (x3) - Unlocked early Saturday
- **Efficiency bonus**: Awarded only on perfect (1.0 correctness) submissions

Maximum possible score: **6.0** (perfect Tier 3 with max efficiency bonus).
