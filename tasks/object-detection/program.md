# autoresearch: object-detection

This is the object-detection variant of the `autoresearch` pattern.

The goal is not to maximize a leaked local metric. The goal is to win the real competition by turning every model change into a traceable decision against the best trustworthy evidence we have.

## Mission

- Win the NorgesGruppen task.
- Primary optimization target: competition score = `0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`.
- Strategic bias: detection improvements are worth about 3x classification improvements.
- Hard anchor: the current server score `0.9051` is the only trusted end-to-end number right now.

## Current Reality

- The old `stratified_split` validation set is leaked and cannot be used for model selection.
- `data-creation/create_clean_split.py` promotes the existing non-overlapping `198/50` holdout into `data-creation/data/clean_split/`.
- The best current path is still `V5 YOLO + DINOv2 classification`, with MarkusNet letterbox ONNX as the side lane.
- Submission slots are precious. Do not auto-submit.

## Setup

Before running the loop, verify or do this work:

1. Confirm the clean split exists at `tasks/object-detection/data-creation/data/clean_split/`.
2. Read `tasks/object-detection/data-creation/data/clean_split/leakage_audit.json` and treat it as the local eval source of truth.
3. Do not trust any result produced from `stratified_split`.
4. Make sure every evaluator, leaderboard, and watcher is pointed at `clean_split` explicitly.
5. Initialize `tasks/object-detection/autoresearch_results.tsv` if missing.
6. Initialize or update experiment provenance tracking before starting a new batch.

## In-Scope Files

Read these first:

- `tasks/object-detection/README.md`
- `tasks/object-detection/data_leakage_report.json`
- `tasks/object-detection/data-creation/create_clean_split.py`
- `tasks/object-detection/data-creation/build_final_dataset.py`
- `tasks/object-detection/vlm-approach/eval_stratified_map.py`
- `tasks/object-detection/vlm-approach/build_val_leaderboard.py`
- `tasks/object-detection/vlm-approach/watch_checkpoints.py`
- `tasks/object-detection/yolo-approach/export_v5_submission.py`

Read these when working on a specific branch of the stack:

- `tasks/object-detection/submission-single-model/run.py`
- `tasks/object-detection/submission-markusnet/run.py`
- `tasks/object-detection/submission-markusnet/run_fast.py`

## Trusted Metrics

Use this ranking of trust:

1. Private server score from competition submissions.
2. Clean validation score on `clean_split`.
3. External-data training metrics that do not touch the leaked local split.
4. Everything measured on `stratified_split`: ignore for selection.

## What You Can Change

- Eval automation, leaderboard generation, checkpoint watchers, export/package flows.
- Clean-val threshold, NMS, and TTA sweeps.
- Experiment tracking and provenance files.
- Training configs and post-processing if they can be evaluated on clean val or justified for server testing.

## What You Must Not Do

- Do not auto-submit to the competition.
- Do not use leaked val metrics for any keep/discard decision.
- Do not start CPU training.
- Do not spend time on blocked paths unless the human explicitly reopens them.

Blocked paths right now:

- MarkusNet pure PyTorch reimplementation.
- Dynamic ONNX export for the vision encoder.
- Vendoring `transformers` into the submission sandbox.
- SLERP model merging in its current form.

## Results Logging

Log every evaluated artifact to `tasks/object-detection/autoresearch_results.tsv`.

Columns:

```tsv
timestamp	experiment_id	model	train_data	eval_data	leakage_status	combined_score	detection_map50	classification_map50	server_score	runtime_sec	compute_hours	submissions_used	submission_ready	status	description
```

Status values:

- `keep`
- `discard`
- `blocked`
- `needs_server_eval`
- `submission_candidate`

## Provenance Contract

Every experiment must carry this minimum schema somewhere machine-readable:

```json
{
  "experiment_id": "...",
  "model": {"name": "...", "checkpoint": "..."},
  "training_data": {"name": "...", "sources": []},
  "eval_data": {"name": "clean_split_50", "path": "..."},
  "leakage_status": "clean|suspect|unknown",
  "compute": {"device": "...", "precision": "..."},
  "metrics": {"det_map50": null, "cls_map50": null, "combined": null, "runtime_sec": null},
  "submission_ready": false
}
```

## The Loop

Loop forever unless interrupted.

1. Sync the state of the clean split, tracker, current best server score, and available checkpoints.
2. If the evaluator stack still points at `stratified_split`, fix that first.
3. Score every new checkpoint or submission artifact on `clean_split`.
4. Append the result and provenance to the tracker.
5. If the artifact is a YOLO checkpoint, run the export path: `best.pt -> ONNX -> submission ZIP`.
6. Run lightweight sweeps on post-processing only after a clean baseline exists.
7. Produce a ranked shortlist for humans, but do not submit automatically.
8. Keep only changes that improve a trusted metric or materially improve automation reliability.
9. If a change only improves leaked metrics, revert it.

## Keep / Discard Rules

- Keep if clean-val combined score improves with acceptable runtime and no new leakage risk.
- Keep if detection improves meaningfully even with flat classification, unless runtime becomes submission-risky.
- Keep if automation reliability improves without harming evaluation correctness.
- Discard if improvement exists only on leaked or untrusted metrics.
- Discard if packaging/export becomes brittle enough to threaten the current safe submission.

## Submission Policy

- The human decides when to spend a submission slot.
- Autoresearch prepares candidates, manifests, provenance, and expected tradeoffs.
- Prefer server submissions for candidates that either:
  - clearly beat current clean-val best, or
  - represent a distinct strategic bet not captured by current trusted offline eval.

## Pareto Frontier

Maintain the frontier on at least these axes:

- `combined_score` vs `runtime_sec`
- `combined_score` vs `compute_hours`
- `server_score` vs `submissions_used`

Suggested command:

```bash
python tools/pareto_frontier.py \
  --input tasks/object-detection/autoresearch_results.tsv \
  --x runtime_sec --x-direction min \
  --y combined_score --y-direction max \
  --label experiment_id \
  --time timestamp
```

## Bottom Line

The correct object-detection autoresearch loop is:

`change -> clean eval -> provenance log -> shortlist -> human submission decision`

Not:

`change -> leaked metric -> false confidence -> wasted submission`
