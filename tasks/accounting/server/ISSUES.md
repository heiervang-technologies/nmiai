# Accounting Agent Issue Tracker

All agents: check this before pushing fixes. Update status when fixing.
Only %2 (master) restarts the server. Agents commit+push and notify %2.

## OPEN (Day 4 priority)

| # | Severity | Family | Issue | Root Cause | Owner |
|---|----------|--------|-------|------------|-------|
| 26 | RED | invoice | Payment reversal creates TWO invoices | LLM ignores action-layer auto-create signal, creates second invoice. Fix deployed (3b6c6f3f) — action returns {COMPLETE:true} — UNCONFIRMED | failure-analysis %5 |
| 25 | RED | invoice | "formation" triggers food VAT detection | Word "formation" falsely matches food signals. vatTypeId=31 applied to 25% standard item. Fix needed: tighten food keyword list | blindspot %3 |
| 24 | YELLOW | project | 4.5/8 — fixed-price project LLM ignores typed tool | LLM uses hourlyRates.fixedRate instead of project.fixedprice. Action-layer intercept deployed (e63eaf13, 9d3da67f) — UNCONFIRMED | advisor %6 |
| 23 | YELLOW | invoice | 6.5/8 — order→invoice flow check 4 fails | Order created but /:invoice conversion needs bank account + invoiceDate. Fix deployed (08bcfdd) — UNCONFIRMED | blindspot %3 |
| 22 | YELLOW | salary | 3/8 — voucher-only salary misses scorer checks | process_salary uses voucher fallback. Scorer may want real /salary/transaction | advisor %6 |
| 21 | YELLOW | planner | "Registrer timer på prosjektet" → project not timesheet | Word "prosjektet" wins over "timer". Need timesheet priority boost or multi-word phrase | all |
| 20 | YELLOW | planner | "Opprett dimensjon Produktlinje" → product not voucher | "Produkt" in dimension name matches product family | all |
| 28 | RED | project | Project lifecycle 4 errors: hourlyRates 422 + incomingInvoice 403 + duplicates | Complex multi-step prompt. hourlyRateModel validation fails (sent "FIXED_HOURLY_RATE" not "TYPE_FIXED_HOURLY_RATE"), incomingInvoice 403, then duplicate retries. Mock too permissive — does not validate hourlyRateModel enum. Prompt: "Execute complete project lifecycle for System Upgrade Greenfield" | failure-analysis |
| 29 | YELLOW | planner | Portuguese salary misrouted to invoice | "pagamento" matches invoice keywords, ties with "salario" for salary. Invoice wins by priority. FIXED: added PT multi-word salary keywords (27c735f6) | harness |
| 30 | YELLOW | planner | Portuguese cost_analysis misrouted to project | "projeto" in prompt matches project keywords (2.0) beating cost_analysis (1.0). FIXED: added PT multi-word cost_analysis keywords (27c735f6) | harness |

## FIXED (confirmed or deployed)

| # | Severity | Family | Issue | Fix | Commit |
|---|----------|--------|-------|-----|--------|
| F27 | YELLOW | planner | French supplier invoice misclassified as "invoice" | Already fixed by prior keyword update. Planner now correctly routes "facture fournisseur" to supplier. Confirmed via harness. | 27c735f6 |
| F29 | YELLOW | planner | Portuguese salary misrouted to invoice | Added PT multi-word salary keywords | 27c735f6 |
| F30 | YELLOW | planner | Portuguese cost_analysis misrouted to project | Added PT multi-word cost_analysis keywords | 27c735f6 |
| F20 | RED | dimension | 0/8 — dimension on BOTH postings | Strip from bank accounts | de513af |
| F19 | RED | annual_close | Nynorsk misclassified as cost_analysis | Priority swap + keyword | 9aab175 |
| F18 | RED | supplier | /incomingInvoice 422: missing externalId | Added externalId + hardened voucher fallback | f7f1309, faad591 |
| F17 | RED | invoice | 0% VAT sent as id=31 (15% food) | Hard override for exempt keywords + vatPercentage field | b8f9814, 9144b7b, c4cfd240 |
| F16 | RED | invoice | Payment classified as "customer" | Added payment keywords in 6 languages | 0887e92 |
| F15 | RED | invoice | Payment used excl-VAT amount | Auto-correct from invoice outstanding balance | 2c8c589 |
| F14 | RED | all | Server --reload causes 0/8 during restarts | Removed --reload, only master restarts | operational |
| F13 | RED | invoice | order/:invoice 422 missing invoiceDate | Auto-inject + bank account setup + dueDate +30 | 08bcfdd |
| F12 | YELLOW | employee | userType=NO_ACCESS, no email from PDF | PDF limit 6000 chars, placeholder email, never downgrade | ee2e5c9 |
| F11 | YELLOW | travel | Redundant /expense/:deliver (wrong endpoint) | Redirect /expense/ → /travelExpense/ | cdc1a2f |
| F10 | YELLOW | travel | isForeignTravel always False | Derive from countryCode | 9450b5d |
| F9 | YELLOW | product | VAT cascade tries all types (13 errors) | Limited to 3 attempts | 9450b5d |
| F8 | YELLOW | employee | Department defaults to first | Create from prompt | 9450b5d |
| F7 | YELLOW | department | departmentNumber collision with default | Auto-assign next available | 5c1e462 |
| F6 | YELLOW | cost_analysis | Activity creation empty body | Guard + GENERAL_ACTIVITY via /activity | 0c7a553 |
| F5 | YELLOW | invoice | dueDate = invoiceDate | Default +30 days | 12e99cb |
| F4 | YELLOW | timesheet | Separate tools instead of combined | System prompt routing | 2c8c589 |
| F3 | YELLOW | invoice | Credit note on pre-populated invoice | Must create customer+invoice first | d39dc8f |
| F2 | YELLOW | invoice | /invoice/:send missing sendType | Auto-inject sendType=EMAIL | 8ff2072 |
| F1 | YELLOW | customer | "cliente" keyword gives customer double score | Removed "client" from customer, dedup | e1ac6097 |

## RULES (hard code guards — prompt hints unreliable)

- **0% VAT**: Action layer forces vatType.id=5 when description has exempt keywords
- **Non-food vatType=31 override**: If vatTypeId=31 but description has no food keywords → 0% exempt
- **No --reload**: Server restarted manually by master (%2) only
- **Payment amount**: Uses invoice outstanding balance, not prompt amount
- **Payment reversal**: Action layer creates full chain (customer→invoice→+payment→-payment)
- **userType**: Never downgrades to NO_ACCESS, generates placeholder email
- **Travel redirect**: /expense/ → /travelExpense/ automatically
- **Dimension stripping**: freeAccountingDimension* stripped from bank account postings
- **Fixed-price intercept**: hourlyRates PUT with fixedRate → redirect to typed action
- **Fresh sandbox**: ALWAYS create entities first, never assume pre-existing data
- **vatPercentage required**: System prompt requires BOTH vatTypeId AND vatPercentage on order lines

## Score History

| Time | Score | Rank | Notes |
|------|-------|------|-------|
| Day 2 evening | 45.8 | #40 | Before major refactor |
| Day 3 pre-reboot | 71.0 | #21 | Peak before PC freeze |
| Day 3 post-fixes | 71.5 | #22-26 | After 20+ fixes, 300 submissions |
