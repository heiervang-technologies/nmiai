# Tripletex - Scoring

## Verification Method

Scoring uses **field-by-field verification** against expected results in the sandbox.

### Example: Create Employee

Total possible points: **10**

| Field | Points |
|-------|--------|
| Found (entity exists) | 2 |
| firstName | 1 |
| lastName | 1 |
| email | 1 |
| admin | 5 |

## Score Calculation

### Correctness

Correctness is **normalized to 0-1** based on points earned vs. points possible.

```
correctness = points_earned / total_points
```

### Tier Multiplier

Tasks are released in tiers with increasing multipliers:

| Tier | Multiplier | Availability |
|------|------------|-------------|
| Tier 1 | x1 | Available now |
| Tier 2 | x2 | Unlocked early Friday |
| Tier 3 | x3 | Unlocked early Saturday |

### Efficiency Bonus

The efficiency bonus is awarded **only on perfect submissions** (correctness = 1.0).

Criteria:
- **Fewer API calls** = higher bonus
- **Zero 4xx errors** = higher bonus
- Bonus can be **up to 2x** the tier multiplier

### Final Score Formula

```
score = correctness * tier_multiplier + efficiency_bonus
```

Maximum score: **6.0** (perfect Tier 3 submission with maximum efficiency bonus).

## Score Retention

The **best score per task** is kept across all submissions.

## Benchmark Recalculation

Benchmarks are recalculated **every 12 hours**.

## Rate Limits

### Verified Teams

| Limit | Value |
|-------|-------|
| Concurrent submissions | 3 |
| Per task per day | 5 |

### Unverified Teams

| Limit | Value |
|-------|-------|
| Concurrent submissions | 1 |
| Per task per day | 2 |
