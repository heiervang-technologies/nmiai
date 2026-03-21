# Agent Instructions - NM i AI 2026

## What This Is

Competition entry for NM i AI 2026 (Norwegian Championship of AI).
- **3 tasks**, 69 hours, **deadline Sunday March 22 15:00 CET**
- Prize pool: 1,000,000 NOK
- Platform: https://app.ainm.no
- Team: The Vector Space

## Quick Orientation (read in order)

1. **`RULES.md`** - ALL agent rules (MANDATORY READ)
2. `AGENTS.md` - current team roster and pane assignments
3. `README.md` - project structure, chain of command diagram
4. Your task's `tasks/<task>/README.md` - task-specific strategy
5. Your GitHub tracking issue - current deliverables and progress
6. `docs/official/` - competition docs mirrored from app.ainm.no

## Rules Summary (see RULES.md for full details)

- **Double-enter** after every tmux message send (race condition)
- **XML agent tags** in all inter-agent messages
- **Only masters** use `say` command
- **Agents stay in their session** - no cross-session reassignment
- **uv** for Python (not pip/conda)
- **GPU only** for training (never CPU)
- **GPT-5.4** for LLM tasks (not GPT-4o)
- **Don't ask for confirmation** - just do it
- **No false dichotomies** - run all viable tracks in parallel
- **Commit and push** to main frequently
- **No model weights** in git (.pt, .pth, .onnx, .safetensors)
- **Register yourself**: `bash register-agent.sh "name" "role" "desc"`
- **Late game = exploitation** - stay within trust region

## Tracking Issues

| Issue | Purpose |
|-------|---------|
| [#6](https://github.com/heiervang-technologies/nmiai/issues/6) | **Master tracker** |
| [#2](https://github.com/heiervang-technologies/nmiai/issues/2) | Object Detection |
| [#3](https://github.com/heiervang-technologies/nmiai/issues/3) | Accounting |
| [#4](https://github.com/heiervang-technologies/nmiai/issues/4) | Astar Island |

## Communication

Master orchestrator: **pane %5** (nmai-master session).

```bash
# Send message to another pane
director send %TARGET 'message'
sleep 0.5
tmux send-keys -t %TARGET Enter
sleep 0.3
tmux send-keys -t %TARGET Enter
```

## The Three Tasks

| Task | Metric | Our Score | Leader |
|------|--------|-----------|--------|
| **Object Detection** | mAP@0.5 (70% det + 30% cls) | 98.1 | ~100 |
| **Accounting** (PRIORITY) | Field correctness × tier × efficiency | 45.8 | 100.0 |
| **Astar Island** | KL divergence 0-100 | 93.6 | ~100 |

## Commit Guidelines

- Conventional commits (`feat:`, `fix:`, `docs:`)
- Commit early, commit often, push to main
- Do NOT commit model weights or large binaries
