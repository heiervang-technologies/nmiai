# Astar Island Breakthrough Analysis

This note answers the higher-level question: why are we around raw score `71` while the winning teams are around weighted `118` / raw `80-85`, and what is the most plausible breakthrough?

## Executive summary

The main gap is probably **not** that we completely misunderstand the automaton's mean behavior. The bigger issue is that we are still getting the **sharpness** of the distribution wrong on the cells that matter most.

Our current models are decent at predicting the dominant class family on many cells, but they are too often **under-dispersed** on volatile frontier cells. The key statistic points directly at this:

- about `4487` cells (`11%`) have priors that are too narrow: `H_prior < H_gt - 0.3`
- those cells have about **3x worse weighted KL**

Under entropy-weighted KL, that is exactly the failure mode that kills score.

The likely breakthrough is therefore:

1. keep a good **mean-distribution model** for `E[p | x]`
2. add a separate **entropy / concentration model** for how uncertain each cell should be
3. calibrate the mean prediction to the target entropy per cell via temperature or Dirichlet concentration

In short: **predict both the distribution and its uncertainty level**.

## Quantifying the gap

The official score formula is:

```text
score = 100 * exp(-3 * weighted_kl)
```

So:

- raw `71` means weighted KL about `0.114`
- raw `80` means weighted KL about `0.074`
- raw `85` means weighted KL about `0.054`

That means getting from `71` to `85` is not a cosmetic improvement. It requires cutting weighted KL by about **half**.

Easy cells are already mostly solved:

- ocean is deterministic
- mountain is deterministic
- many quiet plains / forest interiors are low entropy

So the missing gain must come from the **dynamic frontier band**:

- settlement growth halos
- coastal trade / port cells
- collapse / rewilding frontier
- conflict-adjacent cells

Those are exactly the cells where entropy is high and the score weights are largest.

## What we are likely missing

### 1. We predict the mean better than the variance

Current modeling is mostly trying to estimate:

```text
q(x) ≈ E[p | features(x)]
```

But the score punishes us heavily if `q(x)` is too sharp when the true `p(x)` is broad.

That is our current failure signature:

- correct class family, wrong confidence
- too much mass on the top class
- not enough mass on secondary modes like `Empty/Settlement/Forest` or `Settlement/Port/Empty`

This is especially damaging because weighted KL emphasizes exactly those broad cells.

### 2. We are missing a heteroscedastic uncertainty model

The automaton is clearly **heteroscedastic**:

- initial `Port` cells have mean entropy about `1.17`
- initial `Settlement` cells have mean entropy about `1.04`
- initial `Forest` cells are much lower at about `0.60`
- initial `Plains` cells are lower still at about `0.51`

So there is strong structure in uncertainty. Some regimes are consistently broad, some consistently sharp.

But a single prior-strength rule or one global smoothing constant cannot represent that.

### 3. We are under-modeling regime boundaries

From the automaton analysis, the highest-entropy cells are not random. They sit on recognizable regime boundaries:

- near settlements and ports
- on coastal frontier cells
- at forest edge near civ
- in dense civ clusters where growth and collapse compete
- on cells that can plausibly split between `Empty`, `Settlement`, `Forest`, and sometimes `Port` or `Ruin`

The score gap is therefore probably not about discovering one more rule like “ports like coasts.” We already know that. The gap is about knowing **how uncertain** the coast-frontier cell should be.

## Can entropy be learned from features?

Yes.

I ran a quick sanity check using the existing `predictor.py` spatial features and a small entropy regressor.

Result:

- baseline using only initial-type mean entropy: LORO entropy MAE about `0.334`
- HistGBT regressor on current basic spatial features: LORO entropy MAE about `0.289`

That is not good enough yet, but it proves the key point:

- **entropy is learnable from local map features**
- the current feature set already contains real signal
- the remaining weakness is that the features and calibration target are still too crude

So the answer to the core question is: **yes, we should explicitly learn the entropy level per cell**.

## The actual breakthrough

The most plausible breakthrough is a **two-stage probabilistic model**:

### Stage A: Predict the mean distribution

Use the best available predictor to estimate the class mixture:

```text
m(x) ≈ E[p | x]
```

This can come from:

- bucket / empirical Bayes model
- spatial model
- recursive / denoising model
- ensemble of several predictors

The goal here is class ranking and support.

### Stage B: Predict the uncertainty level

Train a second model to estimate either:

- target entropy `H*(x) ≈ E[H(p) | x]`, or better
- optimal concentration / temperature `T*(x)` or `k*(x)`

This second model should use:

- initial terrain type
- distance to civ
- coast / ocean adjacency
- forest edge / forest interior
- local civ density
- settlement/port neighbor counts
- model-predicted entropy
- top-1 probability and top-1/top-2 margin
- disagreement between different predictors
- current-round observation variance if queries exist

This is effectively a **meta-model for calibration**, not for argmax.

## How to use the entropy model

There are two strong ways to operationalize it.

### Option 1: Temperature-match the entropy

Take a base distribution `m(x)` and apply a per-cell temperature:

```text
q(x) = softmax(log(m(x)) / T(x))
```

Choose `T(x)` so that:

```text
H(q(x)) ≈ H_hat(x)
```

Interpretation:

- `T > 1`: flatten an overconfident prediction
- `T < 1`: sharpen an underconfident prediction

This preserves the base model's ranking while fixing confidence.

### Option 2: Predict Dirichlet concentration

Represent the cell with:

```text
alpha(x) = k(x) * m(x)
```

where:

- `m(x)` is the mean class distribution
- `k(x)` is the concentration / confidence

Small `k(x)` means broad and uncertain. Large `k(x)` means concentrated and confident.

This is especially clean because the current observation update already wants a Dirichlet-style prior.

## Why this is the right target under the score

The score is entropy-weighted KL, so the most important cells are exactly the ones where being overconfident hurts most.

That means the best model is not just “best argmax” or even “best mean probabilities.” It is the model with the best **per-cell calibration of sharpness**.

This is why the `4487 too narrow cells` statistic is so important. It identifies the dominant residual failure mode.

If we fix those cells, we are fixing the part of the loss surface that actually matters.

## What winning teams are plausibly doing

I do not think the winners are winning by one tiny trick. They are probably doing some version of this:

1. a stronger conditional distribution model than simple buckets
2. better support constraints on impossible classes
3. much better calibration on volatile cells
4. heavier repeat-query concentration on high-entropy frontier regions

The likely differentiator is that they are closer to modeling the simulator as a **stochastic local process**, not just a transition table.

But even without fully learning the simulator, the biggest practical gain available to us is still the calibration layer:

- mean model says what can happen
- entropy model says how broad it should be

That is the minimum change that directly targets the observed scoring bottleneck.

## Proposed implementation plan

### 1. Build a calibration dataset from out-of-fold predictions

For every historical ground-truth cell:

- generate out-of-fold prediction `m_hat(x)` from the base model
- record features from initial grid and local neighborhood
- record diagnostic features from the prediction itself

Targets:

- `H_gt(x)`
- or optimal `T*(x)` minimizing `H(p) * KL(p || temp(m_hat, T))`
- or optimal `beta*(x)` in a flattening mixture

```text
q_cal = (1 - beta) * m_hat + beta * u_local
```

where `u_local` is not pure uniform, but a plausible broad prior over supported classes.

### 2. Use a richer feature set for uncertainty than for mean prediction

Useful additional features:

- local heterogeneity of initial terrain
- settlement cluster edge indicator
- coastal civ indicator
- local conflict proxy: civ density without coast support
- distance to multiple structure types
- model disagreement across predictors
- predicted top-1 margin
- whether the cell is in a known ambiguous regime: `plains near civ`, `coastal civ`, `forest edge near civ`, `old settlement`

### 3. Switch query allocation to expected entropy reduction

Once we predict `H_hat(x)`, query value should be:

```text
value(viewport) ≈ sum over cells of H_hat(x) * expected_reduction_in_uncertainty(x)
```

That should outperform hotspot scoring based only on civ adjacency.

### 4. Evaluate by the official objective only

Every change should be judged by:

- leave-one-round-out weighted KL
- number of cells with `H_pred << H_gt`
- calibration curves by regime

The new headline metric should be something like:

```text
underdispersion_rate = fraction of cells with H_pred < H_gt - 0.3
```

because that is currently the damaging failure mode.

## My best current conclusion

The breakthrough is **not** “find a better deterministic rule set.”

The breakthrough is:

**learn a per-cell uncertainty model and use it to calibrate prediction sharpness.**

More concretely:

1. predict mean distribution `m(x)`
2. predict entropy / concentration `H(x)` or `k(x)`
3. reshape `m(x)` to the correct sharpness
4. use that same uncertainty model to drive repeat-query allocation

If we want to close the gap from raw `71` to raw `80-85`, this is the most plausible path.

## Recommended next build

The next serious version should be an **entropy-calibrated ensemble**:

1. Start from the best mean predictor or ensemble.
2. Train an out-of-fold entropy / temperature meta-model.
3. Apply per-cell temperature scaling or Dirichlet concentration adjustment.
4. Re-run LORO and specifically measure whether the `too narrow` cell count drops.

If that works, then a stronger simulator-style model becomes worth the additional complexity. But right now, the biggest missing piece is calibration, not another mean predictor.
