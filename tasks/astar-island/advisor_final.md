# Final Advisor: Top 3 Improvements for the Last 24 Hours

## 1. Replace hard regime classification with per-seed posterior regime inference

Right now [regime_predictor.py](/home/me/ht/nmiai/tasks/astar-island/regime_predictor.py) uses a fairly blunt observation rule in `detect_regime_from_observations()`: it counts settlement/port hits on frontier cells and then jumps to hand-coded regime weights. That is useful, but it throws away too much information.

Action:

- For each regime `r in {harsh, moderate, prosperous}`, score the observed blitz viewport under that regime prior:
  - `log P(obs | r) = sum_over_observed_cells log q_r(cell_class | bucket)`
- Then compute a posterior over regimes:
  - `P(r | obs) propto P(obs | r) * P(r)`
- Do this **per seed**, not once globally for the whole round.

Why this matters:

- [behavior_patterns.md](/home/me/ht/nmiai/tasks/astar-island/behavior_patterns.md) shows regime is the dominant hidden variable after distance to settlement.
- Your current detection only uses a frontier settlement rate. It ignores ports, forests, empties, and the full likelihood structure of the observed viewport.
- With 10 repeated queries on the hottest viewport, you have enough evidence to infer a much better regime posterior than the current threshold logic.

Expected gain:

- Biggest gain on “surprise prosperous” or “surprise harsh” rounds like R14.
- This is probably the single highest-EV improvement because it changes the whole prior, not just the queried cells.

Implementation note:

- Keep the current 3-regime tables.
- Only replace the regime-weight selection logic.
- Add a small floor when computing log-likelihood so rare classes do not dominate numerically.

## 2. Add frontier-specific entropy calibration by distance band and initial type

Your tau improvement from `2 -> 10` is good, but it is still one global smoothing knob over a very non-uniform error surface.

The critical miss zone from your description matches exactly what [behavior_patterns.md](/home/me/ht/nmiai/tasks/astar-island/behavior_patterns.md) says:

- plains / forest at distance `4-7` from settlements are still dynamic,
- but much less deterministic than deep interior,
- and they contribute a lot of avoidable KL when the prior is too sharp.

Action:

- Add a second-stage calibration layer on top of the regime prior with **bucket-specific temperature or convex mixing with uniform**.
- Use at least these buckets:
  - initial type in `{plains, forest, settlement}`
  - nearest-settlement distance bands `{0, 1-2, 3-4, 5-7, 8+}`
  - coast flag
- Fit calibration from historical ground truth by minimizing in-sample weighted KL or matching bucket entropy.

Recommended simple rule if you need something tonight:

- `dist 0-2`: keep sharp
- `dist 3-4`: mild flattening
- `dist 5-7`: strongest flattening
- `dist 8+`: collapse hard to empty/forest depending on initial type

Why this matters:

- [behavior_patterns.md](/home/me/ht/nmiai/tasks/astar-island/behavior_patterns.md) shows that distance >= 12 is deterministic, but `4-7` is exactly the “long tail frontier” where naive priors stay overconfident.
- Weighted KL punishes those moderate-entropy cells more than static ones.

Expected gain:

- Smaller than regime posterior inference, but probably easier and safer.
- This should directly attack the 3x KL hotspot you already identified.

## 3. Split query policy into two explicit modes: regime-identification seeds and payoff seeds

Your R13 result strongly suggests that blitzing the hottest viewport works when the round structure is favorable. The next step is to make query allocation more deliberate instead of uniformly “10 hottest per seed”.

Action:

- Use the first `1-2` queries per seed only for **regime evidence** on the hottest frontier viewport.
- After that, rank seeds by expected score impact:
  - frontier area size,
  - number of initial settlements,
  - uncertainty under the current regime posterior,
  - amount of coast-adjacent civ where ports can flip mass.
- Then spend the remaining queries disproportionately on the top `2-3` seeds where repeated samples reduce the most weighted KL.

Why this matters:

- [behavior_patterns.md](/home/me/ht/nmiai/tasks/astar-island/behavior_patterns.md) says entropy is highest on initial settlement cells, ports, and immediate frontier cells.
- Not all seeds are equally valuable. If one seed has much larger frontier / civ mass, repeated samples there are worth more than continuing to sample a low-entropy seed.
- Once regime posterior is confident for a seed, further queries should be used only if they materially sharpen local cell distributions.

Concrete heuristic:

- Seed value score:
  - `V_seed = 2.0 * frontier_cells + 1.5 * initial_settlement_count + 1.0 * coastal_frontier_cells + 1.0 * posterior_entropy(regime)`
- Allocate:
  - `2` recon queries to every seed,
  - remaining budget to top-value seeds for repeat sampling on the best hotspot(s).

Expected gain:

- This is the best way to turn the current blitz strategy into a more robust round-winning policy, especially on high-weight rounds where one bad seed can drag the whole average.

## Priority order

If you only have time for one change:

1. Regime posterior from observation likelihoods

If you have time for two:

1. Regime posterior
2. Frontier-specific calibration for `dist=4-7`

If you have time for three:

1. Regime posterior
2. Frontier calibration
3. Seed-level adaptive query allocation

## What I would not spend the last 24 hours on

- More generic ML models without explicit regime handling
- Global temperature tuning only
- Adding more map features unrelated to distance/coast/frontier density

The current evidence says the score is being won or lost on:

- hidden round regime,
- frontier calibration,
- and spending repeats where entropy-weighted KL is concentrated.
