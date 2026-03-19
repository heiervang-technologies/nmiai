# Astar Island Solver Feedback

Current issue: the solver uses one global uniform Laplace prior (`+0.5` to every class in every observed cell). That is too blunt. It is still too sharp in volatile settlement zones, and too conservative in stable forest/plains cells.

## 1. Recommended smoothing values

If you keep scalar Laplace smoothing, use this instead of a single global `0.5`:

| Zone | Recommended pseudo-count | Effect after 1 observation of one class |
|---|---:|---:|
| Hot / high-entropy | `0.75` | observed class gets about `31.8%` |
| Cold / low-entropy | `0.10` | observed class gets about `68.8%` |

Optional middle tier if you want smoother transitions:

| Zone | Pseudo-count | 1-observation posterior |
|---|---:|---:|
| Warm / medium-entropy | `0.30` | observed class gets about `46.4%` |

Practical zoning:

- `hot`: initial settlement/port cells, cells within Manhattan distance `<= 2` of an initial settlement, and coastal frontier cells.
- `cold`: forest interior, plains interior, and cells with distance `>= 4` from any settlement and no coast adjacency.
- `warm`: everything between those two.

Why these values:

- `0.5` everywhere is a poor compromise. In cold cells it keeps predictions too flat, and in hot cells it still lets one sample swing the cell too hard.
- `hot = 0.75` is heavy enough that one observation does not dominate, but not so heavy that 2-3 repeated observations are ignored.
- `cold = 0.10` lets stable cells become confident quickly, which matches forest interior and quiet plains much better.

## 2. Use an informed Dirichlet prior instead of uniform Laplace

Yes. This is the bigger improvement.

Uniform Laplace wastes mass on implausible classes:

- non-mountain cells still get mountain mass,
- inland cells still get port mass,
- initial forests and initial settlements get the same prior even though their transition behavior is obviously different.

Preferred formulation:

```python
posterior = counts[y, x] + tau[y, x] * prior_mean[y, x]
pred[y, x] = posterior / posterior.sum()
```

Where:

- `prior_mean[y, x]` is a 6-class probability vector from the initial state and local features.
- `tau[y, x]` is the prior strength.

Recommended prior strengths:

- `tau_hot = 4.0`
- `tau_warm = 2.0`
- `tau_cold = 0.75`

Those `tau` values are the Dirichlet version of “heavy vs light smoothing”. They are better than uniform Laplace because the prior mass goes to plausible classes instead of being sprayed evenly across all 6 classes.

Suggested prior means:

- Initial settlement cell:
  `m = [0.12, 0.48, 0.08, 0.22, 0.10, 0.00]`
- Initial port cell:
  `m = [0.08, 0.18, 0.55, 0.11, 0.08, 0.00]`
- Initial forest interior:
  `m = [0.18, 0.05, 0.01, 0.04, 0.72, 0.00]`
- Initial plains / empty interior:
  `m = [0.80, 0.10, 0.01, 0.04, 0.05, 0.00]`
- Coastal non-settlement cell:
  `m = [0.72, 0.10, 0.08, 0.04, 0.06, 0.00]`

These do not need to be perfect. Even rough informed priors are better than uniform because they stop the model from donating large probability to impossible or near-impossible outcomes.

## 3. Other prediction-building improvements

### A. Put hard support constraints on impossible classes

- Initial mountain: hardcode mountain.
- Initial ocean: hardcode empty.
- Non-mountain cells: give mountain `0` before the final floor.
- Non-coastal cells: port should be near `0` before the final floor.

This matters because the current uniform prior leaks too much mass into mountain and port.

### B. Use feature-conditioned priors for unobserved cells

Right now unobserved non-static cells are effectively uniform. That is weak.

For unobserved cells, back off to a prior based on:

- initial cell type,
- distance to nearest settlement,
- coast adjacency,
- forest-edge vs forest-interior,
- local settlement density within radius 2 or 3.

That alone should beat uniform by a lot.

### C. Pool evidence across similar cells

Many cells are observed only once because the current viewport plan mostly covers the map rather than repeating hot zones. Use empirical Bayes pooling:

- build buckets like `(initial_type, dist_to_settlement_bin, coast_flag, forest_edge_flag)`,
- estimate pooled class frequencies from all observed cells in the bucket,
- use that pooled distribution as `prior_mean` for each cell in the bucket.

This gives you much better priors without needing historical labels.

### D. Define “hotness” from features, not only raw settlement distance

Distance to settlement is the best first signal, but I would score instability from several features:

- distance to nearest settlement,
- whether the cell itself is an initial settlement,
- coast adjacency,
- forest edge,
- number of settlements within radius 3.

Then map that instability score to `tau` continuously instead of using a brittle binary threshold.

### E. Spend the last queries on repeated samples of hot regions

This is not strictly smoothing, but it directly affects prediction quality. The README explicitly says repeated queries on the same region estimate the true distribution. Because the score upweights high-entropy cells, repeated samples near settlement clusters are probably worth more than full-map single coverage.

## Bottom line

- If you stay with plain Laplace: use about `0.75` in hot zones and `0.10` in cold zones, with `0.30` as an optional middle tier.
- Better approach: replace uniform Laplace with a feature-informed Dirichlet prior and vary total prior strength by cell instability.
- Biggest gains are likely from informed priors, support constraints, and pooling across similar cells, not from fine-tuning one global pseudo-count.
