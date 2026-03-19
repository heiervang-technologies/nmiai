# NM i AI 2026 - Team Heiervang Technologies

Competition entry for the Norwegian Championship of AI (NM i kunstig intelligens) 2026.

**Competition:** March 19 18:00 CET - March 22 15:00 CET (69 hours)
**Platform:** https://app.ainm.no
**Team:** The Vector Space (invite code: 54AFB021)

## Tasks

We are competing in 3 tasks, each ~33% of the overall score:

| Task | Description | Tracking Issue |
|------|-------------|---------------|
| **Object Detection** | Detect & classify grocery products on shelves (mAP@0.5) | [#2](https://github.com/heiervang-technologies/nmiai/issues/2) |
| **AI Accounting Agent** | Solve Tripletex accounting tasks via /solve endpoint | [#3](https://github.com/heiervang-technologies/nmiai/issues/3) |
| **Astar Island** | Predict Norse simulator world state (KL divergence) | [#4](https://github.com/heiervang-technologies/nmiai/issues/4) |

Master tracker: [#6](https://github.com/heiervang-technologies/nmiai/issues/6)

## Repository Structure

```
nmiai/
  competition-rules.md      # Full competition rules
  research-notes.md         # General research findings
  tasks/
    object-detection/       # NorgesGruppen product detection
      README.md             # Task spec and strategy
      yolo-approach/        # YOLOv8/YOLO26 training and submission
      vlm-approach/         # DINOv2 classification pipeline
      sam3-approach/        # ONNX inference prototyping
      data-creation/        # Dataset augmentation scripts
    accounting/             # Tripletex AI accounting agent
      README.md             # Task spec and strategy
    astar-island/           # Norse island prediction
      README.md             # Task spec and strategy
      solver.py             # Round solver
  docs/                     # Kickoff event notes and transcripts
```

## Agent Architecture

This project is managed by a multi-agent system running in tmux:

| Agent | Pane | Role |
|-------|------|------|
| **nmai-master** | %7 | Master orchestrator - coordinates all agents |
| **master-object-detection** | %8 | Object detection task lead |
| **master-accounting** | %9 | Accounting task lead |
| **master-astar-island** | %10 | Astar island task lead |

Each master agent may spawn sub-agents for parallel work (training, research, data processing).

### Agent Communication Protocol

Agents communicate via tmux. Due to a known issue with `director send`, always follow up with Enter:
```bash
director send %TARGET 'your message'
sleep 0.5
tmux send-keys -t %TARGET Enter
```

Report back to master using:
```bash
tmux-tool send %7 '<agent id="YOUR_ID" role="YOUR_ROLE" pane="%YOUR_PANE">message</agent>'
sleep 0.5
tmux send-keys -t %7 Enter
```

### Onboarding a New Agent

1. Read this README for project overview
2. Read `competition-rules.md` for full rules
3. Read your task's README under `tasks/<task>/README.md`
4. Check your tracking issue for current strategy and status
5. Use `uv` for all Python package management (not pip/conda)
6. Commit and push to main regularly
7. Update your tracking issue body when strategy changes

## Tech Stack

- **Python** (managed with `uv`)
- **YOLOv8 / RT-DETR** for object detection
- **DINOv2** for product classification
- **FastAPI** for accounting endpoint
- **Claude / LLM** for accounting task parsing

## Key Rules

- Code must be MIT licensed and repo public before deadline
- AI assistants (Claude, ChatGPT, Copilot) are permitted
- No code sharing between teams
- Vipps verification required for prizes

## License

MIT
