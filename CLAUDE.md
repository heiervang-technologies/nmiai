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

We communicate between agents via tmux panes. Each agent runs in its own pane.

### Master orchestrator: pane %2

All agents report to the master orchestrator in pane %2.

### How to send messages to other agents

Use `tmux-tool send` or `director send` to paste text into another pane. **CRITICAL: The message will NOT auto-submit.** You MUST send Enter separately afterward.

**Recommended pattern (most reliable):**
```bash
director send %TARGET 'your message here'
sleep 0.5
tmux send-keys -t %TARGET Enter
sleep 0.3
tmux send-keys -t %TARGET Enter
```

**Alternative using tmux-tool:**
```bash
tmux-tool send %TARGET 'your message here'
sleep 0.5
tmux send-keys -t %TARGET Enter
sleep 0.3
tmux send-keys -t %TARGET Enter
```

### KNOWN ISSUE: Enter key not submitting

There is a race condition where Enter is sent before the paste completes, causing messages to sit in the input buffer unsubmitted. **Always send Enter twice with a short delay to be safe.** This is tracked in [agent-tools#7](https://github.com/heiervang-technologies/agent-tools/issues/7).

If you see `[Pasted text #N +X lines]` in a pane but the agent is not processing, the message was not submitted. Send Enter manually:
```bash
tmux send-keys -t %TARGET Enter
```

### Reporting to master

Use the multi-agent XML protocol:
```bash
tmux-tool send %2 '<agent id="YOUR_ID" role="YOUR_ROLE" pane="%YOUR_PANE">Your message</agent>'
sleep 0.5
tmux send-keys -t %2 Enter
sleep 0.3
tmux send-keys -t %2 Enter
```

### Finding your pane ID
```bash
tmux-tool current
# or
echo $TMUX_PANE
```

### Listing all agents
```bash
director list
tmux-tool list
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
