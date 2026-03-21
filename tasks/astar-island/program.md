# autoresearch: astar-island

This is the Astar Island variant of the `autoresearch` pattern.

The loop is split in two:

- offline, honest validation on completed rounds
- live, conservative round execution with a safe-best submission path

## Mission

- Win the Astar Island task.
- Primary optimization target: lower leave-one-round-out mean weighted KL.
- Live objective: deploy the best validated predictor on every remaining round, without overwriting a stronger safe submission.

## Current Reality

- Current best model: `regime_predictor.py` with Bayesian regime posterior, regime-specific priors, structural zeros, and floor `1e-6`.
- The trusted validation signal is leave-one-round-out CV on completed round ground truth only.
- In-sample benchmarks are not decision-grade.
- `auto_watcher.sh` is the current safe-best live policy and should not be overridden casually.

## Setup

Before starting the loop:

1. Fetch any missing ground truth from completed rounds.
2. Confirm `tasks/astar-island/ground_truth/` is current.
3. Confirm `tasks/astar-island/autoresearch_results.tsv` exists.
4. Confirm `eval_system.py` or equivalent honest CV path is using only held-out rounds for evaluation.
5. Treat `auto_watcher.sh` as production until a challenger proves itself out-of-sample.

## In-Scope Files

Read these first:

- `tasks/astar-island/README.md`
- `tasks/astar-island/regime_predictor.py`
- `tasks/astar-island/eval_system.py`
- `tasks/astar-island/auto_watcher.sh`
- `tasks/astar-island/query_runner.py`
- `tasks/astar-island/eval_final.md`
- `tasks/astar-island/advisor_final.md`
- `tasks/astar-island/gemini_advice.md`

## Trusted Metrics

Use this ranking of trust:

1. Leave-one-round-out CV mean weighted KL.
2. Per-round held-out KL consistency and variance.
3. Live competition score after submission.
4. In-sample results: ignore for keep/discard.

## What You Can Change

- Predictor logic.
- Query selection logic.
- Honest CV harnesses.
- Logging, diagnostics, and frontier tracking.

## What You Must Not Do

- Do not ship a change based only on in-sample gains.
- Do not override the live safe-best pipeline unless the challenger beats it by more than `5%` on held-out CV.
- Do not spend live rounds on experiments that have no honest offline support.

## Results Logging

Log every serious offline or live evaluation to `tasks/astar-island/autoresearch_results.tsv`.

Columns:

```tsv
timestamp	experiment_id	model	cv_weighted_kl	cv_rounds	live_score	queries_used	compute_minutes	submissions_used	override_safe_best	status	description
```

Status values:

- `keep`
- `discard`
- `safe_best`
- `challenger`
- `manual_review`

## Required Live Loop

Autoresearch should preserve this production rhythm:

1. Between rounds: fetch GT, rebuild priors, rerun honest CV, update frontier.
2. Round opens: ingest round metadata and announce.
3. At about 60 minutes after open: spend the query budget according to the live policy.
4. At about 30 minutes before close: submit with the current safe-best predictor.
5. After scoring: compare live score to offline expectation and diagnose misses.

## The Loop

Loop forever unless interrupted.

1. Fetch newly completed ground truth.
2. Rebuild or refresh the predictor.
3. Run leave-one-round-out CV.
4. Record mean weighted KL, per-round breakdowns, and variance.
5. If a challenger beats safe-best by more than `5%` on held-out CV, mark it as a manual promotion candidate.
6. Otherwise keep the live watcher unchanged.
7. After each live round, log the realized score and compare it to the CV-implied expectation.

## Keep / Discard Rules

- Keep changes that improve held-out mean weighted KL and do not materially worsen variance.
- Keep changes that specifically improve prosperous-round behavior without hurting moderate/harsh rounds too much.
- Discard changes that only help in-sample.
- Discard changes that weaken the watcher safety guarantees.

## Known Bottlenecks

- Prosperous rounds remain the hardest regime.
- Regime detection and overlay settings changed recently and still need honest CV.
- Round-to-round variance is still too high.

## Pareto Frontier

Maintain the frontier on at least these axes:

- `cv_weighted_kl` vs `compute_minutes`
- `live_score` vs `submissions_used`
- `cv_weighted_kl` vs `queries_used`

Suggested command:

```bash
python tools/pareto_frontier.py \
  --input tasks/astar-island/autoresearch_results.tsv \
  --x compute_minutes --x-direction min \
  --y cv_weighted_kl --y-direction min \
  --label experiment_id \
  --time timestamp
```

## Bottom Line

The correct astar autoresearch loop is:

`new GT -> honest CV -> challenger decision -> safe live deployment`

Not:

`tweak predictor -> look at in-sample score -> overwrite production watcher`
