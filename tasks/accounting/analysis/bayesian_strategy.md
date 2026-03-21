# Bayesian Submission Strategy — Accounting

## The Problem (Multi-Armed Bandit with Code Fixes)

- **30 tasks**, each scored independently. Best-ever score kept.
- **We don't choose** which task we get. Platform assigns, biased toward less-attempted tasks.
- **300 submission budget**, ~24h remaining.
- **Current score: 45.8/100**
- **Rate limit**: 10 per task per day, 3 concurrent.

This is NOT a standard bandit. We can't pull specific arms. Instead:
1. Fix agent code → improves expected reward for a family
2. Submit → platform picks a task (biased toward unseen/low-attempt)
3. Observe score → update beliefs
4. Repeat

## Current Posterior (Belief State)

| Family         | Tier | MaxPts | Runs | P(success) | E[score] | Status     |
|----------------|------|--------|------|------------|----------|------------|
| department     | T1   | 2.0    | 0    | 0.80*      | 1.6      | UNSEEN     |
| product        | T1   | 2.0    | 0    | 0.80*      | 1.6      | UNSEEN     |
| employee       | T1   | 2.0    | 1    | 0.10       | 0.2      | BROKEN     |
| customer       | T1   | 2.0    | 1    | 0.15       | 0.3      | BROKEN     |
| invoice        | T2   | 4.0    | 0    | 0.50*      | 2.0      | UNSEEN     |
| travel_expense | T2   | 4.0    | 2    | 0.30       | 1.2      | PARTIAL    |
| project        | T2   | 4.0    | 1    | 0.10       | 0.4      | BROKEN     |
| supplier       | T2   | 4.0    | 1    | 0.15       | 0.6      | BROKEN     |
| salary         | T2   | 4.0    | 2    | 0.70       | 2.8      | WORKING    |
| timesheet      | T2   | 4.0    | 0    | 0.40*      | 1.6      | UNSEEN     |
| voucher        | T3   | 6.0    | 1    | 0.05       | 0.3      | BROKEN     |

*Prior based on code review — tools exist but untested against scorer.

## Optimal Strategy: Fix-Submit-Observe Cycles

### CRITICAL INSIGHT: Fix code BEFORE burning submissions

Every submission on a broken family is wasted. The platform biases toward
unseen families, so our first ~15-20 submissions will hit department, product,
invoice, timesheet. If those are broken, we burn budget for zero score.

### Phase 1: CODE FIXES (no submissions) — Target: 1-2 hours

Fix these bugs (ordered by expected impact × ease):

1. **voucher** — dateFrom/dateTo null on /ledger/posting. T3 = 6.0 pts max. HIGHEST ROI.
2. **travel_expense** — rateType.id missing on perDiemCompensation. Must be type=travel not employee expense.
3. **supplier** — /incomingInvoice doesn't have 'account' field. Wrong API schema.
4. **project** — /timesheet/entry needs activity (not null). Use /activity endpoint.
5. **employee** — 0 API calls in logs. Possibly parser not routing to create_employee.
6. **customer** — voucher postings need customer.id in the posting object.
7. **salary** — employmentType field doesn't exist in API. Fix field name.

### Phase 2: EXPLORATION BURST — 30 submissions

After code fixes, submit 30 times. Platform will heavily bias toward the 4 unseen
families (department, product, invoice, timesheet). Expected distribution:

- ~8 unseen family hits (2 each for dept/product/invoice/timesheet)
- ~22 spread across seen families with updated code

**After this burst**: Analyze results. Update posterior. Identify which families
are still scoring poorly.

### Phase 3: TARGETED FIXES — Based on Phase 2 data

Look at scores per family. For any family scoring < 50% of max:
- Read the log, find what the scorer checked that we missed
- Fix the specific field/logic
- This is where Bayesian updating shines — we now have real scorer feedback

### Phase 4: EXPLOITATION — 100 submissions

With fixes applied, submit 100 more. By now most families should be functional.
Platform will distribute evenly since we've attempted everything.

Expected: ~3 submissions per family. Best-score-kept means even 1 good run locks it in.

### Phase 5: FINAL POLISH — 70 submissions

Focus on efficiency bonus for families scoring correctness=1.0.
Minimize API calls (remove unnecessary GETs, skip discover_sandbox for simple tasks).
Perfect correctness DOUBLES the tier score.

### Reserve: 100 submissions

Keep 100 in reserve for:
- Tier 3 tasks (opening "early Saturday" = may not be open yet)
- Hot fixes for families that score 0 despite code being correct
- Final hour burst

## Submission Cadence

| Time Window     | Submissions | Purpose                        |
|-----------------|-------------|--------------------------------|
| After fixes     | 30          | Exploration burst              |
| +1h analysis    | 0           | Fix based on scorer feedback   |
| After fix #2    | 100         | Main exploitation run          |
| +1h analysis    | 0           | Final polish                   |
| After fix #3    | 70          | Efficiency optimization        |
| Final 2 hours   | 100         | Reserve / emergency            |
| **Total**       | **300**     |                                |

## Key Decision Rules

1. **NEVER submit if >3 families are known broken.** Fix first.
2. **Stop submitting a family** once it hits correctness=1.0 (only need efficiency after that).
3. **If a family scores 0 twice**, stop submitting and deep-debug the logs.
4. **Tier 3 tasks = 3x multiplier.** Every % of correctness is worth 3x. Prioritize.
5. **GET requests are FREE** for efficiency. Keep preflight lookups, cut unnecessary POSTs.

## Expected Outcome

Conservative estimate with this strategy:
- T1 families (4×2.0): ~6.0 pts (75% correctness avg)
- T2 families (6×4.0): ~16.0 pts (67% correctness avg)
- T3 families (1×6.0): ~3.0 pts (50% correctness)
- Efficiency bonuses: ~4.0 pts (on 2-3 perfect families)
- **Total: ~29 pts gain → 45.8 + 29 = ~75**

Optimistic with good fixes:
- **Total: ~40 pts gain → 45.8 + 40 = ~86**
