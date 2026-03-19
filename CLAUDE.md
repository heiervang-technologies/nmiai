# Agent Instructions - NM i AI 2026

## What This Is

Competition entry for NM i AI 2026 (Norwegian Championship of AI). 3 tasks, 69 hours, deadline March 22 15:00 CET. See `competition-rules.md` for full rules.

## Quick Orientation

1. Read `README.md` for project structure and agent roster
2. Read your task's `tasks/<task>/README.md` for task-specific details
3. Check your GitHub tracking issue for current strategy (issues #2, #3, #4)
4. Master tracker: issue #6

## Development

- Use `uv` for all Python (not pip, not conda)
- Commit and push to main frequently
- Update your tracking issue body when strategy changes

## Communication

- Master orchestrator is in pane **%7**
- Report to master: `tmux-tool send %7 '<agent ...>message</agent>'` then `sleep 0.5 && tmux send-keys -t %7 Enter`
- Known bug: `director send` does not auto-submit. Always follow with `sleep 0.5 && tmux send-keys -t %TARGET Enter`

## Commit Guidelines

- Use conventional commits (`feat:`, `fix:`, `docs:`)
- Do NOT commit model weights, datasets, or large binaries (check `.gitignore`)
- Commit early, commit often
