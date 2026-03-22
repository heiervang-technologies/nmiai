# autoresearch: astar-island

This is an experiment to have the LLM autonomously improve an island simulator predictor.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`).
2. **Read the in-scope files**:
   - `tasks/astar-island/README.md` — task context, API, scoring.
   - `tasks/astar-island/regime_predictor.py` — the predictor. This is the file you modify.
   - `tasks/astar-island/eval_cv.py` — leave-one-round-out cross-validation. Read-only unless running validation refinement.
   - `tasks/astar-island/ground_truth/` — completed round data.
3. **Verify ground truth is current**: Check that `ground_truth/` contains data from completed rounds.
4. **Initialize results.tsv** with the header row.
5. **Confirm and go**.

## Experimentation

Each experiment runs leave-one-round-out cross-validation on completed rounds. No live API calls needed — this is pure offline evaluation.

You launch evaluation as: `python tasks/astar-island/eval_cv.py`

**What you CAN do:**
- Modify `regime_predictor.py` — this is the only file you edit. Everything is fair game: regime detection, prior construction, query strategy, tau parameter, port estimation, structural zeros, population modeling, Bayesian updates.

**What you CANNOT do:**
- Modify `eval_cv.py` in the optimization loop. It is read-only.
- Use more than 50 queries per round (competition budget).
- Access future round data during prediction (the CV harness enforces this).

**The goal is simple: get the lowest `mean_wkl`.** This is the mean entropy-weighted KL divergence across held-out rounds. Lower is better.

**Simplicity criterion**: All else being equal, simpler is better. A predictor with fewer special cases that achieves equal wKL is preferred.

## Output format

The evaluator prints:

```
---
mean_wkl:         0.0580
std_wkl:          0.0120
worst_round:      R12
worst_wkl:        0.0890
rounds_evaluated: 15
queries_per_round: 50
```

Extract: `grep "^mean_wkl:" run.log`

## Logging results

Log to `results.tsv` (tab-separated).

```
commit	mean_wkl	std_wkl	status	description
```

Status: `keep`, `discard`, or `crash`.

Example:
```
a1b2c3d	0.058000	0.012	keep	baseline
b2c3d4e	0.055000	0.011	keep	conditional tau by growth rate
c3d4e5f	0.061000	0.015	discard	UNet-style interpolation (worse + higher variance)
```

## The experiment loop

LOOP FOREVER:

1. Look at git state and current best.
2. Modify `regime_predictor.py` with an idea.
3. `git commit`
4. Run: `python tasks/astar-island/eval_cv.py > run.log 2>&1`
5. Read results: `grep "^mean_wkl:\|^worst_round:" run.log`
6. If crashed, check `tail -n 50 run.log`.
7. Record in results.tsv.
8. If mean_wkl improved (lower): keep.
9. If equal or worse: `git reset` back.

**NEVER STOP**: Do not ask the human. You are autonomous. If stuck, re-read the ground truth data, analyze per-round breakdowns, study regime patterns, try different Bayesian priors. The loop runs until interrupted.

## Validation refinement (separate loop)

When running validation refinement instead of optimization:

1. The file you modify is `eval_cv.py`, not the predictor.
2. The goal is to make CV score more predictive of live competition score.
3. Compare CV predictions to actual live scores from past rounds.
4. Adjust fold strategy, weighting scheme, or scoring edge cases.
5. A better evaluator is one where CV rank order matches live rank order.
6. Log refinements to `val_results.tsv` with columns: `commit	correlation	description`.
