# NM i AI 2026 - Competition Rules & Information

> Compiled from app.ainm.no, ainm.no, and nmiai2026.no on 2026-03-19.

## Overview

- **Event:** NM i AI 2026 (Norwegian AI Championship)
- **Organizer:** Astar Technologies AS (Org.nr: 934040147) / Astar Consulting
- **Contact:** erik@astarconsulting.no, +47 932 87 479
- **Duration:** 69 hours, March 19 at 18:00 CET to March 22 at 15:00 CET
- **Platform:** https://app.ainm.no
- **Cost:** Free
- **Communication:** Official Slack workspace (primary), LinkedIn, email

---

## Team Composition & Eligibility

- **Team size:** 1-4 members for main competition (1-5 for pre-competition Grocery Bot)
- **Minimum age:** 15 years at registration
- **One team only:** A person may only be on one team
- **Roster lock:** Once a team makes their first submission in any main task, the roster is locked
- **Vipps verification:** All members must complete Vipps verification (Norwegian BankID) before the deadline to be eligible for prizes
- **U23 eligibility:** All members must be under 23 at competition end date (March 22, 2026)

---

## Tasks (3 Main Tasks)

> **Note:** Grocery Bot was a separate pre-competition (Feb 20 - Mar 16) with its own 10,000 NOK prize. It does NOT count toward the main competition score.

### Task 1: NorgesGruppen Data - Object Detection (Code Upload)

- Detect and classify grocery products on store shelves
- Submit ZIP with `run.py` at root + model weights (max 420 MB)
- Training data: COCO dataset (~864 MB), 248 images, ~22,700 annotations, 356 categories
- Product reference images (~60 MB): 327 products with multi-angle photos
- Sandbox: NVIDIA L4 GPU (24 GB VRAM), pre-installed PyTorch/YOLO/ONNX
- No runtime pip install; restricted imports (no os, subprocess, socket, etc.)
- **Scoring:** 70% detection mAP@0.5 (category ignored) + 30% classification mAP@0.5 (correct category_id)
- Detection-only submissions (category_id: 0) score up to 70%
- Timeout: 300 seconds

### Task 2: Tripletex - AI Accounting Agent (HTTPS Endpoint)

- Build an AI agent executing accounting tasks via the Tripletex API
- Submit HTTPS endpoint URL; platform sends POST to `/solve` with task + credentials
- Fresh sandbox account provisioned per submission
- Task categories: employee management, customer/product registration, invoicing, payments, travel expenses, projects, corrections, departments
- 30 unique task variants across 3 tiers
- **Scoring:** Field-by-field verification, tier multipliers (1x/2x/3x), efficiency bonus for perfect correctness (can double tier score)
- Tier release schedule: Tier 1 at start, Tier 2 early Friday, Tier 3 early Saturday
- Timeout: 300 seconds

### Task 3: Astar Island - Norse World Prediction (REST API)

- Query a Norse civilization simulator, predict final terrain state
- 40x40 grid map, 50 queries total per round (shared across 5 seeds)
- 15x15 viewport per query
- 8 terrain types mapped to 6 prediction classes
- 50-year simulation with phases: Growth, Conflict, Trade, Winter, Environment
- Submit 3D prediction tensor: `prediction[y][x][class]` (probabilities summing to 1.0)
- **Critical:** Never assign 0.0 probability to any class (KL divergence becomes infinite). Use minimum floor ~0.01.
- **Scoring:** Entropy-weighted KL divergence, 0-100 scale. Static cells excluded. Best round score kept.
- Timeout: 60 seconds

---

## Scoring Methodology

- Each task normalized to 0-100 scale
- **Overall score = average of normalized scores across 3 main tasks (~33% each)**
- Grocery Bot was a separate pre-competition and does NOT count toward the main score

---

## Prize Distribution

| Place | Amount |
|-------|--------|
| 1st | 400,000 NOK |
| 2nd | 300,000 NOK |
| 3rd | 200,000 NOK |
| Best U23 team | 100,000 NOK |
| **Total** | **1,000,000 NOK** |

- U23 prize is **combinable** with placement prizes (can win both)
- Prizes split equally among team members, paid individually
- Paid out gross (no tax withheld by organizer)
- Pre-competition Grocery Bot: separate 10,000 NOK prize for top team

---

## Prize Eligibility Requirements

Both required:

1. **Identity Verification:** All team members complete Vipps verification (Norwegian BankID) before deadline
2. **Code Submission:** Teams must make their code repository **public** and submit the URL through the platform before the competition deadline

### Code Repository Requirements
- Must contain source code for all tasks
- Must include inference code and training scripts
- **All code must be open-sourced under the MIT license (or equivalent permissive license)**
- Organizers verify submissions contain "genuine AI/ML work"
- No hardcoded or pre-computed responses designed to game test cases

---

## Permitted & Prohibited

### Permitted
- AI coding assistants (ChatGPT, Claude, Copilot)
- Public models and datasets
- Research papers and open-source libraries
- All common programming languages with API/ML capability
- Google Cloud resources (if approved -- Vipps-verified teams can apply)

### Prohibited
- Sharing code, model weights, or solutions between teams
- Participating on multiple teams
- Circumventing rate limits or attacking infrastructure
- Extracting test data or evaluation logic
- Hardcoding responses to game test cases
- Using false Vipps verification identity
- Exploiting platform bugs
- Reverse-engineering server internals
- Sharing credentials

> "If you are uncertain whether a technique is permitted, contact the organizers before using it."

---

## Enforcement

- Consequences range from warnings to permanent platform bans
- Jury decisions are final and binding
- No formal appeals process for main competition (compressed timeline)
- Teams failing code verification lose prize eligibility; next eligible team moves up

---

## Google Cloud Support

- Available to Vipps-verified teams who apply through the platform
- Provides: @gcplab.me account, dedicated GCP project, no credit limits
- Access to Gemini models, Cloud Run, Vertex AI, etc.
- Recommended region: europe-north1

---

## Key Dates

| Date | Event |
|------|-------|
| Feb 20 - Mar 16 | Grocery Bot pre-competition |
| Mar 19, 18:00 CET | Main competition kickoff |
| Mar 19 | Opening event at MESH Youngstorget, Oslo (livestreamed) |
| Mar 22, 15:00 CET | Competition deadline (all submissions + code repo + Vipps verification) |
| Mar 31 | Tripletex sandbox accounts expire |

---

## Additional Resources

- **MCP Server:** https://mcp-docs.ainm.no/mcp (for Claude integration)
- **Rules page:** https://app.ainm.no/rules
- **Docs:** https://app.ainm.no/docs
- **Tasks:** https://app.ainm.no/tasks
- **Leaderboard:** https://app.ainm.no/leaderboard
- **FAQ:** https://nmiai2026.no/en/faq
- **Manifest:** https://ainm.no/en/manifest

---

## Data Privacy & IP

- Submissions analyzed for quality assurance and anti-cheating
- Personal data handled per Norwegian GDPR regulations
- Teams retain code ownership
- Organizers gain non-exclusive, royalty-free right to reference participation and results in public communications

---

## Sponsors

- **Platinum:** Astar Consulting, NorgesGruppen, Tripletex
- **Gold:** DNV, Miles, UiA
- **Strategic:** Digital Norway, Founders Hub, Google Cloud, MESH Community, Oda, Tek Norge, KI-NORGE
