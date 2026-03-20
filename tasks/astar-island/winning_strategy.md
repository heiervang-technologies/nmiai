# Winning Strategy for Closing the Last 3-7 Points

## Executive Call

The fastest path is **Option 2**, but implemented as a **latent round-template posterior**, not just a hard 3-way regime label.

Use early observations to infer the current round's global growth curve, then switch the whole predictor to the matching prior family, then spend the rest of the budget on repeated hotspot sampling. This should be the primary next build.

Do **not** spend the next iteration on global temperature tuning alone.

## Ranked Recommendation

1. **Regime detection from observations**: highest expected gain.
2. **Use observations to correct specific cells**: important, but only after the regime posterior is updated.
3. **Per-cell / per-bucket calibration**: useful as a second-order refinement once the right regime/template is selected.
4. **Better global temperature scaling**: lowest ROI.

## Why This Is the Right Lever

- The current no-observation prior already scores `83.5`, so the remaining gap is probably not a missing local rule like ports/coasts/forest persistence.
- The biggest unresolved variable is the **hidden round state**: settlement survival changes from `1.8%` to `42.2%`, and that shift is consistent across all 5 seeds inside a round.
- The round effect changes both **amplitude** and **decay rate**. That means `P(settle) != f(distance) * g(round)`. A single scalar temperature cannot fix that.
- Your own analysis already shows the failure mode: the pooled prior is decent, but the round-specific curve shape is wrong.
- Because the latent regime is **shared by the whole round**, an early query on one seed gives information that helps all 5 seeds. That is much higher leverage than treating observations as purely local cell corrections.

## What To Build Next

### 1. Replace one pooled prior with a small library of round templates

Do not use one historical average prior.

Instead, precompute priors for either:

- all **8 historical rounds** as templates, or
- a hierarchical version: **8 templates** with a **3-regime prior** on top.

Why 8 templates, not only 3 regimes:

- the prosperous/moderate/harsh split is real,
- but the decay shape still varies meaningfully within a regime,
- and your analysis shows `A` and `B` are nearly independent.

So the practical object should be:

```text
q_prior(x) = sum_t w_t * q_t(x)
```

where `t` is a historical round template and `w_t` is the current posterior over templates.

### 2. Make the first queries explicitly about template inference

Use the first `8-12` queries to estimate the round template, not to maximize map coverage.

Recommended scouting policy:

- Pick the `2-3` seeds with the highest dynamic score from initial-state heuristics.
- Query windows rich in cells with `dist_to_civ in [1, 4]`.
- Force a mix of **inland** and **coastal** windows.
- Include at least `2` repeated queries on the single hottest inland window and `2` repeated queries on the hottest coastal window.

The goal is to estimate these round-level diagnostics quickly:

- **initial settlement survival**
- **growth at distance 1-2 from civ**
- **growth at distance 3-5 from civ**
- **coastal redirect into ports** on cells with `2+` ocean neighbors
- **ruin fraction** on old-settlement / near-civ cells

This is important: use both near and mid-distance buckets. If you only estimate total settlement activity, you will miss the rounds where reach changes even when near-field intensity looks similar.

### 3. Infer a posterior over templates, then predict with the mixture

After each scouting query, update:

```text
w_t propto prior_t * likelihood(observations | template_t)
```

Use simple bucketed likelihoods. You do not need a fancy model for this first version.

Start with buckets like:

- initial settlement cell
- plains/forest, non-coastal, `dist_to_civ = 1-2`
- plains/forest, non-coastal, `dist_to_civ = 3-5`
- coastal near-civ cells with `ocean_neighbors >= 2`
- initial forest near civ

Then build the prior as the posterior-weighted mixture over templates.

This is the cleanest way to exploit the biggest uncovered signal in the problem.

### 4. Spend the remaining budget on repeated hotspot sampling

Once the round posterior is reasonably concentrated, use the remaining `38-42` queries for **repeat sampling**, not broad coverage.

Recommended split after scouting:

- `60-70%` of the remaining budget on repeating the top `4-6` hot windows
- `30-40%` on secondary windows for coverage / backstop

Target windows with the highest:

- predicted entropy
- regime uncertainty
- frontier density near civ
- coast-trade ambiguity

Avoid spending late queries on:

- deep ocean belts
- mountain-heavy regions
- plains / forest interior far from civ

One-off observations mostly help regime inference. Repeated observations are what improve the actual posterior on the high-weight cells.

## How To Use Observations Once You Have Them

Option 4 matters, but it should be **conditional on Option 2**.

Use observations to correct specific cells with an informed Dirichlet update:

```text
posterior = counts + tau(x) * q_prior(x)
```

with `tau(x)` tied to prior entropy / instability.

Good starting values:

- hot cells: `tau = 4.0`
- warm cells: `tau = 2.0`
- cold cells: `tau = 0.75`

Hot cells are:

- initial settlements / ports
- cells within `<= 2` of civ
- coastal frontier cells

Cold cells are:

- forest interior
- plains interior
- far-from-civ non-coastal cells

Do not let one observation flip a hot cell too aggressively. The score punishes under-dispersed mistakes more than broad calibrated predictions.

## Why Option 1 Is Not The Main Path

Tuning `T=1.2` to `1.1` or `1.3` is unlikely to recover the full gap.

Why:

- the round shift is not just overconfidence vs underconfidence,
- it changes the whole distance-decay profile,
- and ports / ruins / forest reclaim move with that profile.

Global temperature is a good cleanup knob after template inference, not the primary breakthrough.

## Why Option 3 Is Also Secondary

Per-cell calibration is directionally right, but without current-round regime inference it still averages over the wrong latent process.

So per-cell temperature scaling should be treated as:

1. regime/template-specific, and
2. a refinement on top of the correct template mixture.

If you apply it before fixing the latent round effect, it will mostly smooth the historical average.

## Minimal Next Build

If there is only time for one serious upgrade before the next submission cycle, do this:

1. Train and save **8 historical round templates** using the current bucket feature system.
2. Add a **template-posterior inference** step from early observations.
3. Predict with the **posterior-weighted template mixture**.
4. Change query allocation to **scout first, then repeat hot windows**.
5. Keep the current Dirichlet observation update, but make `tau` depend on cell hotness.

## Bottom Line

The best next move is **not** finer temperature tuning.

The best next move is to treat the active round as a hidden global latent variable, infer it quickly from a small number of high-information observations, and then use the rest of the budget on repeated frontier sampling under the inferred template mixture.

In short:

- **Primary bet:** Option 2
- **Implementation shape:** latent round-template posterior
- **Secondary lever:** Option 4 on repeated hot cells
- **Cleanup only:** Options 3 and 1
