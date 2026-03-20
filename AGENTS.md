# Agent Roster - NM i AI 2026

Last updated: March 20, ~11:00 CET (Day 2)

## Master

| Agent | Pane | Session | Role |
|-------|------|---------|------|
| **nmai-master** | %7 | nmiai | Master orchestrator. Coordinates all agents, manages issues, tracks progress. |

## Object Detection Team

| Agent | Pane | Role | Status |
|-------|------|------|--------|
| **master-object-detection** | %8 | Task lead. Orchestrates training, submissions, git. | 3 ONNX models ready to submit |
| yolo-approach | %22 | YOLOv8x training | Complete (200 epochs, mAP50=0.802) |
| vlm-approach | %18 | DINOv2 + Qwen3.5 classification | Pipeline ready |
| yolo26-exploration | %23 | RT-DETR + YOLO11x/26x ONNX export | Complete (both exported) |
| data-creation | %24 | Dataset augmentation | V3 complete (2565 imgs) |

## Accounting Team

| Agent | Pane | Role | Status |
|-------|------|------|--------|
| **master-accounting** | %9 | Task lead. Building FastAPI /solve endpoint. | Switching to OpenAI, unblocked |
| oracle | %27 | Research and support | Available |

## Astar Island Team

| Agent | Pane | Role | Status |
|-------|------|------|--------|
| **master-astar-island** | %10 | Task lead. Solver development, round submissions. | Auto-watcher running, R6 submitted |
| GPT 5.4 (Codex) | %21 | Strategy advisor | Active |
| GPT 5.4 (Visual) | %16 | Visual analysis of ground truth | Active |

## How to Join

If you are a new agent being onboarded:
1. Read `CLAUDE.md` for full instructions
2. Read your task's `tasks/<task>/README.md`
3. Check your tracking issue (see CLAUDE.md for issue numbers)
4. Report to master at pane %7 when ready
