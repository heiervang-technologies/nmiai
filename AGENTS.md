# Agent Roster - NM i AI 2026

## Master

| Agent | Pane | Session | Role |
|-------|------|---------|------|
| **nmai-master** | %7 | nmiai | Master orchestrator. Coordinates all agents, manages issues, tracks progress. |

## Object Detection Team

| Agent | Pane | Role | Status |
|-------|------|------|--------|
| **master-object-detection** | %8 | Task lead. Orchestrates training, submissions, git. | Active |
| yolo-approach | %22 | YOLOv8x training on local L4 GPU | Training |
| vlm-approach | %18 | DINOv2 classification pipeline | Pipeline ready |
| yolo26-exploration | %23 | RT-DETR + YOLO11x/26x on Centurion (RTX 3090) | Training |
| data-creation | %24 | Dataset augmentation (V3 delivered) | Complete |

## Accounting Team

| Agent | Pane | Role | Status |
|-------|------|------|--------|
| **master-accounting** | %9 | Task lead. Building FastAPI /solve endpoint. | Implementing |
| oracle | %27 | Research and support | Available |

## Astar Island Team

| Agent | Pane | Role | Status |
|-------|------|------|--------|
| **master-astar-island** | %10 | Task lead. Solver development, round submissions. | Awaiting Round 2 |
| GPT 5.4 (Codex) | %21 | Strategy advisor | Available |

## Other

| Agent | Pane | Session | Role |
|-------|------|---------|------|
| agent-tools | %3 | agent-tools | Fixing director send Enter bug |

## How to Join

If you are a new agent being onboarded:
1. Read `CLAUDE.md` for full instructions
2. Read your task's `tasks/<task>/README.md`
3. Check your tracking issue (see CLAUDE.md for issue numbers)
4. Report to master at pane %7 when ready
