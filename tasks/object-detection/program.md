# autoresearch: object-detection

This is an experiment to have the LLM autonomously improve an object detection submission.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `tasks/object-detection/README.md` — task context, metrics, constraints.
   - `tasks/object-detection/submission-markusnet/run.py` — the submission script. This is the file you modify.
   - `tasks/object-detection/eval_local.py` — the local evaluator. Read-only unless running the validation refinement loop.
4. **Verify data exists**: Check that the validation set exists at `tasks/object-detection/data-creation/data/clean_split/`. If not, tell the human.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**.

## Experimentation

Each experiment evaluates a submission against the local validation set. The submission runs inside the same constraints as the competition: NVIDIA GPU, no network, 360s timeout, 420MB ZIP limit.

You launch evaluation as: `python tasks/object-detection/eval_local.py`

**What you CAN do:**
- Modify `run.py` — this is the only file you edit. Everything is fair game: model loading, preprocessing, postprocessing, NMS thresholds, TTA, confidence filtering, class remapping, ensemble logic.

**What you CANNOT do:**
- Modify `eval_local.py` in the optimization loop. It is read-only. It contains the fixed evaluation.
- Add dependencies that aren't available in the competition sandbox.
- Exceed the 420MB ZIP or 360s runtime constraint.

**The goal is simple: get the highest `combined_map`.** The metric is `0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`. Detection improvements are worth ~2.3x classification improvements.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds fragile complexity is not worth it. Removing code for equal results is a win.

## Output format

The evaluator prints:

```
---
detection_map50:      0.9200
classification_map50: 0.8800
combined_map:         0.9080
runtime_sec:          142.3
zip_size_mb:          380.2
```

Extract the key metric: `grep "^combined_map:" run.log`

## Logging results

Log to `results.tsv` (tab-separated).

```
commit	combined_map	runtime_sec	status	description
```

Status: `keep`, `discard`, or `crash`. Use 0.000000 for crashes.

Example:
```
a1b2c3d	0.908000	142.3	keep	baseline
b2c3d4e	0.912000	155.1	keep	lower NMS threshold to 0.3
c3d4e5f	0.905000	180.2	discard	add horizontal flip TTA
```

## The experiment loop

LOOP FOREVER:

1. Look at git state and current best.
2. Modify `run.py` with an experimental idea.
3. `git commit`
4. Run: `python tasks/object-detection/eval_local.py > run.log 2>&1`
5. Read results: `grep "^combined_map:\|^runtime_sec:" run.log`
6. If grep is empty, it crashed. `tail -n 50 run.log` for the traceback.
7. Record in results.tsv.
8. If combined_map improved: keep the commit.
9. If equal or worse: `git reset` back.

**Timeout**: If a run exceeds 400s, kill it. Treat as failure.

**NEVER STOP**: Do not pause to ask the human. The human may be asleep. You are autonomous. If you run out of ideas, re-read the code, try combining near-misses, try more radical changes. The loop runs until interrupted.

## Validation refinement (separate loop)

When running validation refinement instead of optimization:

1. The file you modify is `eval_local.py`, not `run.py`.
2. The goal is to make local eval more predictive of competition server score.
3. Compare local predictions to known server scores. Minimize the gap.
4. Adjust the val set composition, augmentations, scoring edge cases, or class weighting.
5. A better evaluator is one where local rank order matches server rank order.
6. Log refinements to `val_results.tsv` with columns: `commit	correlation	description`.
