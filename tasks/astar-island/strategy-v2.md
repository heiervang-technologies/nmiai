# Astar Island Strategy V2

## Core Correction

The five seeds do **not** share the same initial map. They differ heavily:

- Pairwise initial-map differences are about `670-721` cells out of `1600`.
- Shared mountain overlap between seed pairs is almost zero (`0-5` cells depending on pair).
- Therefore cross-seed transfer of **position-specific** terrain information is invalid.

This changes the strategy:

- Treat each seed as its own map and its own prediction problem.
- Ocean and mountain should again be treated as static **within a seed**.
- The earlier “mountains change” conclusion was an artifact of comparing different seeds at the same coordinates.
- Any pooled statistics must be conditioned on **features**, not raw coordinates.

## 1. Empirical Bayes Pooling

Yes, but only in a feature-based way, not by position.

The right object is:

```python
bucket = (
    initial_type,
    dist_to_civ_bin,
    coast_flag,
    forest_interior_flag,
    local_civ_density_bin,
)
```

where `civ` means initial settlement or initial port **within the same seed**.

For each bucket, estimate a pooled final-state distribution from observed cells:

```python
p_hat_b = (sum_i counts_bi + alpha0 * m0) / (sum_i n_bi + alpha0)
```

Then use that bucket distribution as the prior mean for any cell in the same bucket:

```python
posterior[y, x] = cell_counts[y, x] + tau[y, x] * p_hat_bucket[y, x]
pred[y, x] = posterior[y, x] / posterior[y, x].sum()
```

Important:

- Pool across seeds only through the bucket, never through `(x, y)`.
- Pool ocean separately from mountain separately from dynamic land cells.
- Ports are too sparse to stand alone; smooth them aggressively toward nearby buckets.

### Empirical base rates from round 1

Using the round-1 observations and grouping by **per-seed initial type**, the pooled empirical transitions are approximately:

- `Settlement -> [Empty 39.3%, Settlement 41.5%, Port 1.3%, Ruin 2.5%, Forest 15.4%, Mountain 0%]`
- `Forest -> [Empty 6.6%, Settlement 16.2%, Port 1.2%, Ruin 1.3%, Forest 74.7%, Mountain 0%]`
- `Plains -> [Empty 77.5%, Settlement 16.2%, Port 1.3%, Ruin 1.1%, Forest 3.9%, Mountain 0%]`
- `Port -> [Empty 12.5%, Settlement 12.5%, Port 37.5%, Ruin 0%, Forest 37.5%, Mountain 0%]`  
  Note: only `8` observed port samples; treat this as noisy.
- `Ocean -> 100% Empty`
- `Mountain -> 100% Mountain`

These are much better default prior means than the earlier hand-tuned ones.

### Entropy of these priors

- Settlement prior entropy is about `1.68` bits.
- Forest prior entropy is about `1.16` bits.
- Plains prior entropy is about `1.04` bits.
- Ocean and mountain are effectively `0` bits.

This matches the intuition that settlement neighborhoods deserve the most query budget.

### Recommended EB hierarchy

Use a hierarchical backoff:

```python
p(bucket_full)
-> p(initial_type, dist_bin, coast_flag)
-> p(initial_type, dist_bin)
-> p(initial_type)
```

Choose the most specific bucket with enough support.

Suggested minimum support:

- use full bucket if `n >= 30`
- else back off one level

For the prior strength, use support-adaptive shrinkage:

```python
tau_b = min(tau_max, c * sqrt(n_b))
```

with a reasonable starting point like:

- `tau_max = 6`
- `c = 0.35`

## 2. Simulation Modeling

Yes, a simplified forward model is worth building. It does not need to be perfect to beat hand-tuned priors.

### Best pragmatic model

Do **not** try to reproduce the hidden simulator exactly first. Build a local conditional model for final class probabilities:

```python
P(final_class | initial_features)
```

This is effectively a supervised surrogate for the 50-year simulator.

### Feature set

Per cell, per seed:

- initial type
- distance to nearest initial settlement or port
- count of civ cells within radius 2 and radius 4
- coast flag
- distance to ocean
- forest interior / edge
- mountain adjacency count
- connected-component size of forest patch
- connected-component size of plains region
- whether cell lies on narrow isthmus / fjord boundary

### Model choices

In order of practicality:

1. Multinomial logistic regression / softmax on engineered features
2. Gradient boosted trees predicting 6-class probabilities
3. Small hand-built cellular forward model only after we have more data

Formula:

```python
q_theta(c | x) = softmax(W phi(x) + b)
```

Train this on observed cells from completed rounds.

### If we want an explicit phase-inspired model

Use a factorized approximation:

```python
P(final) ~= P(growth) * P(conflict | growth_zone) * P(trade | coast,civ) * P(winter | density,terrain) * P(environment | ruin,forest)
```

Examples:

- `growth` raises settlement probability near existing civ cells on plains/forest.
- `conflict` raises ruin probability where multiple civ cells are close.
- `trade` raises port probability for coastal civ cells.
- `winter` raises empty probability in over-dense interior zones.
- `environment` converts ruin/empty back toward forest in forest-heavy regions.

But as an engineering priority, the discriminative surrogate is better first.

## 3. Query Strategy Optimization

The current `9 x 15x15` full-coverage plan is too coverage-heavy.

Facts from the overlap geometry:

- `1225` cells are observed once.
- `350` cells are observed twice.
- `25` cells are observed four times.
- Mean observation multiplicity is only `1.266`.

That is not enough for high-entropy zones when score is entropy-weighted KL.

### Key implication

One observation is worth much less than we hoped on hot cells. If `tau = 4`:

- with `1` observation: data weight is only `1 / (1 + 4) = 0.20`
- with `3` observations: data weight is `3 / 7 = 0.43`
- with `5` observations: data weight is `5 / 9 = 0.56`

So repeat sampling is mandatory where entropy is high.

### Recommended strategy

Use an adaptive two-stage policy.

#### Stage A: cheap reconnaissance

Per seed:

- query `2-4` large viewports to estimate activity and locate dense civ clusters
- compute a hotspot score per candidate viewport

Example hotspot score:

```python
score(v) =
    1.0 * num_initial_civ_in_v
  + 0.5 * num_cells_dist_to_civ_le_2_in_v
  + 0.3 * coast_cells_in_v
  + 0.4 * forest_edge_cells_in_v
```

#### Stage B: exploitation

- spend the remaining budget repeat-sampling the top hot viewports
- target cells with highest predicted entropy and highest uncertainty reduction per query

### Viewport size

Default recommendation:

- keep `15x15` unless the API rewards smaller windows by allowing substantially finer targeting without losing too much hot-cell density

Reason:

- if cost is flat per query, the value per query is mostly “high-entropy cells covered”, not area itself
- smaller viewports only win if hotspot concentration is high enough that a `10x10` can avoid lots of dead cells

Operationally:

- start with `15x15` for scouting
- use `10x10` or even smaller only for exploitation if the hot area is compact and well-localized

### Budget split

Starting point:

- `30-35` queries on hotspot repeat-sampling
- `15-20` queries on reconnaissance / broad coverage

Equivalent hot-share:

- `60-70%` of budget on exploitation is a better baseline than `10 queries/seed` uniform coverage

### Skip boring regions

Yes.

- never query pure ocean belts
- never spend repeat samples on mountain-heavy regions
- rarely query deep plains far from civ
- only lightly sample forest interior unless it is adjacent to civ expansion fronts

## 4. Cross-Seed Transfer

The original cross-seed transfer idea is invalid in its strong form.

What remains valid:

- transfer **feature-conditioned transition statistics** across seeds and rounds
- do **not** transfer coordinate-based predictions across seeds

Valid:

```python
P(final | initial_type=forest, dist_bin=1, coast=0)
```

Invalid:

```python
P(final at x=17,y=22 in seed B) <- observation at x=17,y=22 in seed A
```

So the right mental model is:

- seeds share simulation rules,
- they do not share terrain layout.

## 5. Entropy Weighting / KL Exploitation

Yes. Because scoring is entropy-weighted KL to the true distribution, it is often better to predict a calibrated broad distribution than an aggressive point mass.

### Decision rule

For a high-entropy cell, predict closer to the posterior mean and avoid collapsing mass too fast.

For low-entropy static cells, collapse hard.

This is standard Bayesian decision theory under log loss:

- the Bayes-optimal prediction is the expected true distribution
- under uncertainty about that distribution, shrink toward a calibrated prior

### Practical rule

Let:

- `H_prior(y,x)` be entropy of the current prior mean
- `n(y,x)` be number of observations for the cell

Then choose tau as an increasing function of prior entropy:

```python
tau(y,x) = tau_min + (tau_max - tau_min) * H_prior(y,x) / log2(6)
```

with a starting range like:

- `tau_min = 0.5`
- `tau_max = 6.0`

That automatically gives stronger smoothing in volatile buckets.

### Safer than uniform

For high-entropy cells, “closer to uniform” is better than overconfident wrong spikes, but **bucket-conditioned priors** are better than pure uniform because they stay broad while still respecting the mechanics.

Example:

- for settlement cells, use the empirical settlement prior above, not `[1/6]*6`
- for plains near civ, use a broadened plains-near-civ prior, not plain plains and not uniform

## Recommended V3 Predictor

### Prior builder

Per seed, for each cell:

1. Determine feature bucket.
2. Pull empirical Bayes prior mean from historical completed rounds and earlier observed cells in the current round.
3. Set `tau` from bucket entropy and support.
4. Override static classes:

```python
if initial == ocean: q = [1,0,0,0,0,0]
if initial == mountain: q = [0,0,0,0,0,1]
```

### Posterior update

```python
alpha = tau[y, x] * prior_mean[y, x]
posterior = counts[y, x] + alpha
pred[y, x] = posterior / posterior.sum()
pred = np.maximum(pred, floor)
pred /= pred.sum()
```

with `floor` in the `0.005-0.01` range depending on submission tolerance.

### Bucket seed priors for next round

Until more data exists, use these as default prior means:

```python
PRIOR_OCEAN      = [1.000, 0.000, 0.000, 0.000, 0.000, 0.000]
PRIOR_MOUNTAIN   = [0.000, 0.000, 0.000, 0.000, 0.000, 1.000]
PRIOR_SETTLEMENT = [0.393, 0.415, 0.013, 0.025, 0.154, 0.000]
PRIOR_FOREST     = [0.066, 0.162, 0.012, 0.013, 0.747, 0.000]
PRIOR_PLAINS     = [0.775, 0.162, 0.013, 0.011, 0.039, 0.000]
PRIOR_PORT       = [0.125, 0.125, 0.375, 0.000, 0.375, 0.000]  # low support; shrink heavily
```

Then refine by distance/coast/forest-edge buckets as more labeled rounds accumulate.

## Concrete next steps

1. Build a per-seed feature extractor:
   `initial_type`, `dist_to_civ`, `coast`, `forest_interior`, `local_civ_density`.
2. Build bucketed empirical transition tables from round 1 using only per-seed transitions.
3. Replace hand-tuned priors in `compute_prior()` with these empirical priors.
4. Add support-based backoff for sparse buckets.
5. Replace full-map `9x15x15 per seed` with an adaptive reconnaissance + exploitation policy.
6. Store all future round observations and post-round analyses in a dataset for training a surrogate model.

## Bottom Line

- Cross-seed coordinate transfer is dead; feature-based pooling survives.
- Mountains and ocean are static **within a seed** and should be predicted with near certainty.
- The strongest immediate upgrade is empirical Bayes with **per-seed transition buckets**.
- The strongest query upgrade is to stop buying mostly single observations of hot cells.
- The best medium-term upgrade is a supervised surrogate model `P(final_class | initial_features)` trained on accumulated rounds.
