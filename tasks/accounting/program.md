# autoresearch: accounting

This is an experiment to have the LLM autonomously improve an accounting solver.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`).
2. **Read the in-scope files**:
   - `tasks/accounting/README.md` — task context, scoring, families.
   - `tasks/accounting/server/main.py` — the FastAPI endpoint.
   - `tasks/accounting/server/agent.py` — the LLM agent. This is the primary file you modify.
   - `tasks/accounting/server/actions.py` — the Tripletex API action layer.
   - `tasks/accounting/server/planner.py` — task classification and routing.
   - `tasks/accounting/server/playbooks/*.json` — per-family playbooks.
   - `tasks/accounting/eval_local.py` — the local test harness. Read-only unless running validation refinement.
3. **Verify the endpoint is live**: `curl -s http://localhost:8000/health`
4. **Initialize results.tsv** with the header row.
5. **Confirm and go**.

## Experimentation

Each experiment runs the local test harness against the solver. The harness sends synthetic accounting tasks and checks field correctness.

You launch evaluation as: `python tasks/accounting/eval_local.py`

**What you CAN do:**
- Modify the agent, planner, actions, and playbooks. Everything is fair game: prompt engineering, tool schemas, API call sequences, error recovery, field mapping, family routing.

**What you CANNOT do:**
- Modify `eval_local.py` in the optimization loop. It is read-only.
- Change the endpoint contract (`POST /solve` with the competition schema).
- Exceed the 300s timeout per task.

**The goal is simple: get the highest `solve_rate`.** This is the percentage of test tasks solved correctly, weighted by tier multiplier.

**Simplicity criterion**: All else being equal, simpler is better. A playbook that handles 3 families cleanly beats one that handles 5 families with brittle hacks.

## Output format

The evaluator prints:

```
---
solve_rate:       0.7200
tier1_rate:       0.8500
tier2_rate:       0.6000
tier3_rate:       0.4000
avg_api_calls:    12.3
avg_time_sec:     45.2
families_solved:  9/12
```

Extract: `grep "^solve_rate:" run.log`

## Logging results

Log to `results.tsv` (tab-separated).

```
commit	solve_rate	avg_api_calls	status	description
```

Status: `keep`, `discard`, or `crash`.

Example:
```
a1b2c3d	0.720000	12.3	keep	baseline
b2c3d4e	0.750000	14.1	keep	add invoice payment matching
c3d4e5f	0.710000	18.5	discard	retry loop on 500 errors (worse + slower)
```

## The experiment loop

LOOP FOREVER:

1. Look at git state and current best.
2. Modify the agent/planner/actions/playbooks with an idea.
3. `git commit`
4. Run: `python tasks/accounting/eval_local.py > run.log 2>&1`
5. Read results: `grep "^solve_rate:\|^families_solved:" run.log`
6. If crashed, check `tail -n 50 run.log`.
7. Record in results.tsv.
8. If solve_rate improved: keep.
9. If equal or worse: `git reset` back.

**NEVER STOP**: Do not ask the human. You are autonomous. If you run out of ideas, read the failing test cases, study the Tripletex API docs, analyze error patterns, try different prompt strategies. The loop runs until interrupted.

## Validation refinement (separate loop)

When running validation refinement instead of optimization:

1. The file you modify is `eval_local.py`, not the solver.
2. The goal is to make local eval more predictive of dashboard score.
3. Study real task prompts from `/tmp/accounting-logs/summary.jsonl`.
4. Add task families, edge cases, and field checks that the competition scorer likely uses.
5. A better evaluator is one where local solve_rate tracks dashboard score over time.
6. Log refinements to `val_results.tsv` with columns: `commit	correlation	description`.
