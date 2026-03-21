# autoresearch: accounting

This is the accounting variant of the `autoresearch` pattern.

Unlike the original repo, this is not a closed offline training loop. The real work is submission-driven: observe live tasks, cluster them, harden playbooks, and only then optimize efficiency.

## Mission

- Win the Tripletex task.
- Correctness comes before efficiency.
- Immediate priorities: fix `timesheet` and `travel_expense`, then exploit Tier 3 as it unlocks.

## Current Reality

- The live system is a FastAPI `/solve` endpoint on port `8000`.
- The runtime stack is `Pydantic AI + typed tools + planner + playbooks + actions layer`.
- We do not get competition scores back programmatically.
- Markus must still submit the endpoint URL on the dashboard after reboot events.

## Setup

Before running the loop:

1. Confirm the endpoint is live.
2. Confirm logs are being written to `/tmp/accounting-logs/summary.jsonl` and `/tmp/accounting-logs/*.json`.
3. Read the current playbooks under `tasks/accounting/server/playbooks/`.
4. Initialize `tasks/accounting/autoresearch_results.tsv` if missing.
5. Treat the current tunnel URL as unstable unless Tailscale Funnel is active.

## In-Scope Files

Read these first:

- `tasks/accounting/README.md`
- `tasks/accounting/WINNING_STRATEGY.md`
- `tasks/accounting/TASK_MAPPING.md`
- `tasks/accounting/server/main.py`
- `tasks/accounting/server/planner.py`
- `tasks/accounting/server/agent.py`
- `tasks/accounting/server/actions.py`

Then inspect:

- `tasks/accounting/server/playbooks/*.json`
- `/tmp/accounting-logs/summary.jsonl`

## Trusted Eval Signal

Use this ranking of trust:

1. Dashboard score observed by the human.
2. Proxy clean rate from logs.
3. API success pattern quality by family.

Proxy signal for a likely-good run:

- `api_errors <= 2`, and
- at least one meaningful successful write (`POST`/`PUT`/`DELETE`), and
- no obvious family mismatch or tool-routing failure.

## What You Can Change

- Playbooks.
- Planner classification and routing.
- Typed tool prompts and schemas.
- Action-layer API call sequences.
- Log analysis and reporting scripts.

## What You Must Not Do

- Do not assume file attachments contain usable bytes.
- Do not treat dashboard-free proxy metrics as final truth.
- Do not expose more than the isolated API port.
- Do not optimize efficiency before a family is plausibly correct.

## Results Logging

Log every submission batch analysis to `tasks/accounting/autoresearch_results.tsv`.

Columns:

```tsv
timestamp	batch_id	family	model	proxy_clean_rate	api_errors	successful_writes	dashboard_score	submissions_used	attachment_present	status	description
```

Status values:

- `keep`
- `discard`
- `watch`
- `manual_review`
- `needs_dashboard_check`

## Required Automation

Autoresearch should automate these tasks:

1. Parse `summary.jsonl` after each submission batch.
2. Count clean/error rates by family.
3. Alert when a new family appears, with the prompt text.
4. Flag any family whose clean rate drops below `70%`.
5. Surface top recurring error patterns by family.
6. Track unseen families versus known families.
7. Flag attachment-based tasks, especially if `content_base64` is empty.

## The Loop

Loop forever unless interrupted.

1. Watch the summary log and ingest new entries.
2. Group runs by task family.
3. Update family-level clean rates, error patterns, and task coverage.
4. If a new family appears, create a human-facing alert immediately.
5. If `timesheet` or `travel_expense` remain weak, prioritize those families over already-good ones.
6. When a family appears stable, tighten its playbook and reduce API-call waste.
7. When Tier 3 or attachment tasks appear, route them for manual review before broad automation.
8. Record every batch-level conclusion in the tracker.

## Keep / Discard Rules

- Keep changes that raise proxy clean rate or reduce repeated API failures in a target family.
- Keep changes that remove unnecessary sandbox discovery calls on already-solved families.
- Discard changes that reduce robustness across multilingual prompts.
- Discard changes that only reduce calls but appear to worsen correctness.

## Priority Order

1. `timesheet`
2. `travel_expense`
3. Tier 3 discovery and attachment handling
4. Efficiency bonus work on solved families

## Pareto Frontier

Maintain the frontier on at least these axes:

- `proxy_clean_rate` vs `api_errors`
- `dashboard_score` vs `submissions_used`
- `proxy_clean_rate` vs `submissions_used`

Suggested command:

```bash
python tools/pareto_frontier.py \
  --input tasks/accounting/autoresearch_results.tsv \
  --x api_errors --x-direction min \
  --y proxy_clean_rate --y-direction max \
  --label batch_id \
  --time timestamp
```

## Bottom Line

The correct accounting autoresearch loop is:

`submission batch -> log parse -> family diagnosis -> playbook/tool fix -> next batch`

Not:

`random endpoint edits -> no family tracking -> no idea what improved`
