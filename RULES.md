# Agent Rules - NM i AI 2026

All agents MUST follow these rules. No exceptions.

## 1. Communication

### 1.1 Double-Enter Protocol
When sending messages via `director send` or `tmux-tool send`, the message will NOT auto-submit. Always follow with double-enter:
```bash
director send %TARGET 'message'
sleep 0.5
tmux send-keys -t %TARGET Enter
sleep 0.3
tmux send-keys -t %TARGET Enter
```
Known race condition tracked in [agent-tools#7](https://github.com/heiervang-technologies/agent-tools/issues/7).

### 1.2 XML Agent Tags
ALL inter-agent tmux messages MUST use XML agent tags:
```xml
<agent id="YOUR_ID" role="YOUR_ROLE" pane="%YOUR_PANE">Your message</agent>
```

### 1.3 Misrouted Messages
Pane IDs change after reboots. If you receive a message intended for another agent, REPLY IMMEDIATELY: "Wrong pane - I am [your role], not [intended recipient]."

### 1.4 Voice Output (say command)
Only MASTERS may use the `say` command. Sub-agents report to their master and let them decide what to voice.

## 2. Session Boundaries

### 2.1 Agents Stay in Their Session
Each tmux session is a team. Do NOT reassign agents across sessions. Sessions:
- `nmai-master` - orchestration
- `nmai-object-detection` - object detection team
- `nmai-accounting` - accounting team
- `nmai-astar-island` - astar island team

### 2.2 Chain of Command
Sub-agents report to their session master. Masters report to nmai-master. Do not skip levels unless hard-blocked.

## 3. Development

### 3.1 Python: Use uv
Use `uv` for ALL Python package management. Never pip, never conda.

### 3.2 Training: GPU Only
All training MUST run on GPU/CUDA. Never train on CPU. Check `torch.cuda.is_available()`.

### 3.3 LLM Choice
Use GPT-5.4 for LLM tasks (not GPT-4o). Higher quality matters for task parsing.

### 3.4 Git Discipline
- Commit and push to main frequently. Do NOT let work pile up.
- Use conventional commits (`feat:`, `fix:`, `docs:`).
- Do NOT commit model weights (.pt, .pth, .onnx, .safetensors), datasets, ZIPs, or large binaries.
- Check `.gitignore` before committing.
- No feature branches during competition - push to main.

### 3.5 Tracking Issues
The GitHub issue body IS the canonical plan. Update it when strategy changes.
- Master tracker: [#6](https://github.com/heiervang-technologies/nmiai/issues/6)
- Object Detection: [#2](https://github.com/heiervang-technologies/nmiai/issues/2)
- Accounting: [#3](https://github.com/heiervang-technologies/nmiai/issues/3)
- Astar Island: [#4](https://github.com/heiervang-technologies/nmiai/issues/4)

## 4. Decision Making

### 4.1 Don't Ask for Confirmation
When the path forward is clear, execute immediately. Do not ask "should I do X?" or "want me to proceed?" Every minute counts in a 69-hour competition.

### 4.2 No False Dichotomies
We have massive agent parallelism. Do not frame choices as A or B. Run all viable tracks simultaneously.

### 4.3 Late Game: Exploitation Over Exploration
In the final hours, bias toward proven approaches. Only explore within a trust region - small incremental improvements on what works. No wild gambles that could regress scores.

### 4.4 Advisory Votes
For important strategic decisions, nmai-master holds a 3-party vote (master + GPT-5.4 advisor + Gemini advisor). Unanimous decisions execute immediately.

## 5. Competition Awareness

### 5.1 Metric Gaming on Private Test Sets
All tasks are scored on PRIVATE test sets we never see. Optimize for generalization, not overfitting to val sets. Robust models beat brittle ones.

### 5.2 Specific Metrics
- **Object Detection**: mAP@0.5 IoU (70% detection + 30% classification)
- **Accounting**: field correctness × tier multiplier + efficiency bonus
- **Astar Island**: entropy-weighted KL divergence (0-100)

### 5.3 Submission Discipline
- Never submit something worse than current best (automated rollback).
- Object Detection: 3 submissions/day, 360s timeout, 420MB ZIP, L4 GPU, no network.
- Accounting: unlimited submissions, 300s timeout, fresh sandbox per submission.
- Astar Island: auto-watcher handles submissions. Manual override only within trust region.

### 5.4 Tiebreak
Same score = whoever achieved it first (by submission timestamp) wins higher placement.

## 6. Registration

All agents must register on spawn:
```bash
bash /home/me/ht/nmiai/register-agent.sh "your-name" "your-role" "what you are doing"
```

## 7. Competition Constraints

- Code must be MIT licensed
- Repo must be public before deadline (Sun March 22 15:00 CET)
- AI assistants are explicitly permitted
- No code sharing between teams
- Vipps verification required for prizes
