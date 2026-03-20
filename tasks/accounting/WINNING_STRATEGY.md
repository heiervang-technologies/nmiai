# Winning Strategy - AI Accounting Agent

## The Math

- 30 tasks, best score per task kept forever
- Tier 1 (x1): max 1.0/task, Tier 2 (x2): max 2.0/task, Tier 3 (x3): max 3.0/task
- Efficiency bonus: up to 2x on perfect submissions (max 6.0 for Tier 3)
- Total theoretical max: depends on tier distribution, but ~90-120+ points
- **Bad runs don't hurt us.** We can experiment freely.
- Rate limit: 10 submissions per task per day (verified), 3 concurrent

## Core Insight: Correctness >> Efficiency

The scoring formula is `correctness × tier_multiplier × (1 + efficiency_bonus)`.
If correctness = 0.8, we get 0.8 × multiplier × 1.0 (no efficiency bonus).
If correctness = 1.0, we get 1.0 × multiplier × up to 2.0.

**A perfect solution with bad efficiency scores HIGHER than an imperfect efficient one.**

Example (Tier 2):
- 80% correct, perfect efficiency: 0.8 × 2 × 1.0 = 1.6
- 100% correct, terrible efficiency: 1.0 × 2 × 1.0 = 2.0
- 100% correct, good efficiency: 1.0 × 2 × 1.5 = 3.0
- 100% correct, best efficiency: 1.0 × 2 × 2.0 = 4.0

**Priority order: correctness first, efficiency second.**

## Architecture Decision: Full Agentic (Not Structured Parsing)

### Why NOT structured parsing:
1. We don't know all 30 task types upfront
2. Prompt variants across 7 languages are hard to pre-code
3. Multi-step tasks need dynamic reasoning
4. Can't self-correct when an API call fails
5. Fragile — one wrong field mapping = 0 points

### Why full agentic with tool use:
1. Handles ANY task variant without pre-coding
2. Self-correcting — reads error responses, adjusts, retries
3. Can explore the sandbox (list existing entities, find IDs)
4. Handles all 7 languages natively
5. Handles multi-step reasoning (create customer → create invoice)
6. ONE system handles all 30 tasks
7. Can be improved by enriching the system prompt, not rewriting code

### The tradeoff:
More API calls = worse efficiency bonus. But:
- Efficiency bonus only applies at 100% correctness
- We can optimize later once we know which tasks we're getting
- Even without efficiency bonus, perfect correctness × tier multiplier is a great score

## Model Strategy

### Primary: GPT-5.4 via OpenRouter ($1.75/$14 per M tokens)
- Best reasoning for complex multi-step tasks
- Best at following system prompt instructions precisely
- Use for Tier 2 and Tier 3 tasks where correctness matters most
- Use for all tasks initially until we understand them

### Fallback/Cost-saver: GLM-5 via OpenRouter ($0.72/$2.30 per M tokens)
- 6x cheaper on input, 6x cheaper on output
- Use for Tier 1 tasks once we've confirmed they work
- Use for high-volume testing/discovery phase

### When to use which:
```
Phase 1 (Discovery):     GLM-5 (cheap, need volume)
Phase 2 (Correctness):   GPT-5.4 (accuracy matters)
Phase 3 (Optimization):  GLM-5 for Tier 1, GPT-5.4 for Tier 2/3
```

### Fine-tuning: NOT worth it
- Competition ends Sunday — no time to collect data, train, deploy
- 56 variants per task isn't enough training data anyway
- The system prompt + tool use approach is more flexible
- Fine-tuning locks us into patterns we've seen; agentic adapts to anything

## The Three Phases

### Phase 1: Discovery (NOW → tonight)
**Goal:** See all 30 tasks, log everything, understand the scoring

1. Submit endpoint immediately
2. Log EVERY incoming prompt to a file with timestamp
3. Log every API call and response
4. Run submissions repeatedly to collect task samples
5. Categorize tasks as they come in
6. Note which fields are checked (from score feedback)

**Key tech:** Request/response logging middleware

### Phase 2: Correctness (Tonight → Saturday)
**Goal:** Get 100% correctness on as many tasks as possible

1. Analyze logged prompts — what exactly do they ask for?
2. Enrich the system prompt with specific patterns we've seen
3. Add "known tasks" to the system prompt with exact API flows
4. Fix any edge cases (e.g., date formats, special characters, module activation)
5. Handle file attachments (PDF/image parsing for Tier 3)
6. Test each task type until we get 100%

**Key tech:** Rich system prompt, prompt logging, per-task analysis

### Phase 3: Efficiency (Saturday → Sunday deadline)
**Goal:** Minimize API calls for efficiency bonus on perfected tasks

1. For tasks where we have 100% correctness, analyze API call logs
2. Eliminate unnecessary GET calls (e.g., don't list VAT types if we know id=3)
3. Build "fast paths" — hardcoded sequences for known tasks
4. Switch perfected Tier 1 tasks to GLM-5 to save cost
5. Focus efficiency work on Tier 3 (biggest multiplier, biggest efficiency bonus)

**Key tech:** API call counter, fast-path handlers

## System Architecture

```
POST /solve
  ├── Request logging (save prompt, files, credentials)
  ├── Middleware: check if known fast-path task → direct handler
  └── Agent loop:
      ├── System prompt (Tripletex expert + API knowledge + known patterns)
      ├── User prompt (the task)
      ├── Tools: tripletex_get, tripletex_post, tripletex_put, tripletex_delete
      ├── Max 20 iterations, 240s timeout
      └── Each iteration: LLM decides action → execute → feed result back
  └── Response: {"status": "completed"}
  └── Result logging (API calls made, errors, timing)
```

## System Prompt Strategy (Critical)

The system prompt is our most important asset. It should include:

### 1. API Reference (what we learned from sandbox)
- Exact VAT type IDs (3=25%, 31=15%, 32=12%, 5/6=0%)
- Payment type IDs (from sandbox)
- Module activation (POST /company/salesmodules)
- Invoice creation (direct with inline orderLines)
- Fresh sandbox state (1 employee, 0 customers, 1 department)

### 2. Known Task Patterns (populated from Phase 1 logs)
- "When asked to create employee, here's exactly what to do..."
- "When asked to create invoice, here's the exact call sequence..."
- One-shot examples for each discovered task type

### 3. Critical Rules
- Always set vatType on order lines and products
- Use amountGross for voucher postings
- Activate modules before using module-specific features
- Preserve Norwegian characters exactly
- Dates in YYYY-MM-DD

### 4. Anti-patterns (save API calls)
- Don't list all accounts if you know the account number
- Don't search for entities if you're about to create them
- Don't make GET calls to verify your POST succeeded

## Technology To Build

### Must-have (build today):
1. **Agentic executor** — LLM tool-use loop with Tripletex API tools
2. **Request logger** — save every prompt + response to disk
3. **API call logger** — count and log every Tripletex API call
4. **Model switcher** — env var to swap between GLM-5 and GPT-5.4

### Nice-to-have (build if time):
5. **Fast-path router** — skip agent loop for known simple tasks
6. **Score tracker** — parse score feedback, track per-task progress
7. **Prompt catalog** — deduplicated collection of all seen prompts
8. **Auto-retry** — if score < 1.0, auto-resubmit with GPT-5.4

### Don't build:
- Fine-tuned model (no time, not enough data)
- Custom NLP parser (LLM does this better)
- Database (fresh sandbox each time, nothing persists)
- Complex error recovery framework (LLM handles this natively)

## Cost Estimate

Assuming 500 total submissions over the competition:
- Average 15K tokens per submission (system + prompt + 5 tool rounds)
- GLM-5: 500 × 15K × $0.72/M + 500 × 5K × $2.30/M = $5.40 + $5.75 = ~$11
- GPT-5.4: 500 × 15K × $1.75/M + 500 × 5K × $14/M = $13.13 + $35 = ~$48

**Even all-GPT-5.4 costs under $50.** Cost is not a real constraint. Use the best model.

## Recommendation

**Use GPT-5.4 for everything.** The $37 cost difference is nothing compared to the prize pool (333K NOK for this task). Correctness is king, and GPT-5.4 will get more tasks right.

Switch to GLM-5 ONLY if:
- We need faster iteration during discovery (GLM-5 is faster)
- GPT-5.4 is rate-limited
- We've perfected simple Tier 1 tasks and want to save money

## Timeline

| When | What | Model |
|------|------|-------|
| **NOW** | Build agentic executor + logging, submit endpoint | GLM-5 (fast iteration) |
| **Fri evening** | Analyze logs, enrich system prompt | GPT-5.4 |
| **Fri night** | Correctness push on Tier 1 + Tier 2 | GPT-5.4 |
| **Sat morning** | Tier 3 unlocks — tackle immediately | GPT-5.4 |
| **Sat afternoon** | Efficiency optimization on perfected tasks | Both |
| **Sun morning** | Final push — any remaining tasks | GPT-5.4 |
| **Sun 15:00** | Deadline | — |

## Key Risks

1. **Unknown task types** — mitigated by agentic approach (handles anything)
2. **Module activation** — some tasks may need modules enabled first. Agent must know this.
3. **File attachments (Tier 3)** — PDF/CSV parsing needed. May need to pass file content to LLM.
4. **Cloudflared tunnel instability** — tunnel URL changes on restart. Need to re-submit to dashboard.
5. **Rate limits** — 10/task/day limits iteration speed. Need to be strategic about which tasks to retry.
6. **Year-end closing (Tier 3)** — complex Norwegian accounting knowledge required. This is where GPT-5.4 shines.
