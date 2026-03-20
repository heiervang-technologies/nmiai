# Agent Roster - NM i AI 2026

Last updated: March 20, ~13:45 CET (Day 2, post-reboot)

## Master

| Agent | Pane | Session | Role |
|-------|------|---------|------|
| **nmai-master** | %2 | nmiai | Master orchestrator. Coordinates all agents, manages issues, tracks progress. |

## Object Detection Team

| Agent | Pane | Role | Status |
|-------|------|------|--------|
| **master-object-detection** | %3 | Task lead. Orchestrates training, submissions, git. | 3 ONNX models ready to submit |

## Accounting Team

| Agent | Pane | Role | Status |
|-------|------|------|--------|
| **master-accounting** | %4 | Task lead. FastAPI /solve endpoint via cloudflared. | Endpoint live, needs to submit and iterate |

## Astar Island Team

| Agent | Pane | Role | Status |
|-------|------|------|--------|
| **master-astar-island** | %5 | Task lead. Solver development, round submissions. | R7 active, auto-watcher needs restart |

## How to Join

If you are a new agent being onboarded:
1. Read `CLAUDE.md` for full instructions
2. Read your task's `tasks/<task>/README.md`
3. Check your tracking issue (see CLAUDE.md for issue numbers)
4. Report to master at pane %2 when ready
