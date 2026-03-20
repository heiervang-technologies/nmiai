# Agent Instructions - NM i AI 2026

## What This Is

Competition entry for NM i AI 2026 (Norwegian Championship of AI).
- **3 tasks**, 69 hours, **deadline Sunday March 22 15:00 CET**
- Prize pool: 1,000,000 NOK
- Platform: https://app.ainm.no
- Team: The Vector Space

## Quick Orientation (read in order)

1. This file - agent protocols and dev rules
2. `AGENTS.md` - current team roster and pane assignments
3. `README.md` - project structure and repo layout
4. Your task's `tasks/<task>/README.md` - task-specific strategy and spec
5. Your GitHub tracking issue - current deliverables and progress
6. `docs/official/` - complete competition docs mirrored from app.ainm.no
7. `competition-rules.md` - full rules

## Tracking Issues

| Issue | Purpose |
|-------|---------|
| [#6](https://github.com/heiervang-technologies/nmiai/issues/6) | **Master tracker** - overall status |
| [#2](https://github.com/heiervang-technologies/nmiai/issues/2) | Object Detection strategy |
| [#3](https://github.com/heiervang-technologies/nmiai/issues/3) | Accounting strategy |
| [#4](https://github.com/heiervang-technologies/nmiai/issues/4) | Astar Island strategy |

**The issue body IS the canonical plan.** Update it when your strategy changes.

## The Three Tasks

| Task | Type | Key Metric | Deadline |
|------|------|-----------|----------|
| **Object Detection** (NorgesGruppen) | Upload ZIP with run.py + model | mAP@0.5 (70% detect + 30% classify) | Sun 15:00 |
| **AI Accounting** (Tripletex) | HTTPS /solve endpoint | Field correctness x tier multiplier | Sun 15:00 |
| **Astar Island** | REST API predictions | KL divergence 0-100 | Sun 15:00 |

## Development Rules

- **Python**: Use `uv` for everything (not pip, not conda)
- **Git**: Commit and push to main frequently. Do NOT let work pile up
- **Large files**: Do NOT commit model weights (.pt, .pth, .onnx), datasets, or ZIPs. Check `.gitignore`
- **Docs**: Keep your task README and tracking issue up to date

## Communication Protocol

### Master orchestrator: pane %7

Report to master:
```bash
tmux-tool send %7 '<agent id="YOUR_ID" role="YOUR_ROLE" pane="%YOUR_PANE">Your message</agent>'
sleep 0.5
tmux send-keys -t %7 Enter
```

### Known bug: director send does not auto-submit
Always follow any `director send` with:
```bash
sleep 0.5
tmux send-keys -t %TARGET Enter
```

### When to report to master
- Task completed or milestone reached
- Blocker encountered
- Score received from a submission
- Strategy change needed
- Round started/ended (Astar Island)

## Commit Guidelines

- Use conventional commits (`feat:`, `fix:`, `docs:`)
- Include meaningful descriptions
- Commit early, commit often
- Push to main (no feature branches during competition)

## Competition Constraints

- Code must be MIT licensed (LICENSE file exists)
- Repo must be public before deadline
- AI assistants are explicitly permitted
- No code sharing between teams
- Vipps verification required for prizes
