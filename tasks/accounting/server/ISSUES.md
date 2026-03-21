# Accounting Agent Issue Tracker

All agents: check this before pushing fixes. Update status when fixing.

## OPEN

| # | Severity | Family | Issue | Root Cause | Owner |
|---|----------|--------|-------|------------|-------|
| 20 | RED | dimension | 0/8 — dimension voucher missing vatType + dimension on both postings | Auto-add vatType=3 on expense, dimension ONLY on expense posting | 9aab175 |
| 19 | YELLOW | annual_close | 0/8 — Nynorsk monthly closing misclassified as cost_analysis | Swapped priority: annual_close=98 > cost_analysis=97 | 9aab175 |
| 17 | YELLOW | invoice | 6.5/8 — "Crie um pedido" order->invoice flow missing | Agent goes straight to invoice instead of creating order first. Fix deployed (12e99cb) but unconfirmed. | blindspot %3 |
| 16 | YELLOW | project | 4.5/8 — fixed-price project uses wrong tool | LLM used create_project + manual invoice instead of create_fixed_price_project_invoice. Fix deployed (ec4b8d1) but unconfirmed. | blindspot %3 |
| 15 | YELLOW | salary | 3/8 — voucher-only salary misses scorer checks | process_salary uses voucher fallback. Scorer may want real /salary/transaction. Needs PATH A attempt. | advisor %6 |

## FIXED (confirmed by dashboard)

| # | Severity | Family | Issue | Fix | Commit |
|---|----------|--------|-------|-----|--------|
| 18 | RED | supplier | 0/8 — /incomingInvoice 422: missing externalId | Added externalId to orderLines + vendorId extraction from invoiceHeader | f7f1309 |
| 14 | RED | invoice | 0% VAT sent as id=31 (15% food) | Hard override in action layer for exempt keywords | b8f9814 |
| 13 | RED | invoice | Planner duplicate keyword bug — product won over invoice | Dedup keywords at load time | b8f9814 |
| 12 | RED | invoice | Payment classified as "customer" not "invoice" | Added payment keywords in 6 languages | 0887e92 |
| 11 | RED | invoice | Payment used excl-VAT amount instead of incl-VAT | Auto-correct from invoice outstanding balance | 2c8c589 |
| 10 | YELLOW | employee | 11/14 — userType=NO_ACCESS, no email from PDF | PDF limit 3000->6000, never downgrade userType | ee2e5c9 |
| 9 | YELLOW | travel | 4.5/8 — redundant /expense/:deliver call (wrong endpoint) | Redirect /expense/ to /travelExpense/ | cdc1a2f |
| 8 | YELLOW | travel | isForeignTravel always False, isDayTrip wrong | Derive from countryCode | 9450b5d |
| 7 | YELLOW | product | VAT cascade tries all types (13 errors avg) | Limited to 3 attempts | 9450b5d |
| 6 | YELLOW | employee | Missing department defaults to first | Create department from prompt | 9450b5d |
| 5 | RED | all | 0/8 — server --reload causes 1-2s downtime | Removed --reload, manual restarts only | operational |
| 4 | YELLOW | cost_analysis | Activity creation fails (empty body) | Guard against None/empty, activity via /activity | 0c7a553 |
| 3 | YELLOW | invoice | Due date = invoice date (same day) | Default +30 days | 12e99cb |
| 2 | YELLOW | timesheet | Separate tools instead of register_timesheet_and_invoice | System prompt reinforced combined tool routing | 2c8c589 |
| 1 | YELLOW | cost_analysis | Playbook says DON'T create activities but scorer expects them | Updated playbook to create GENERAL_ACTIVITY | manual edit |

## RULES (hard code guards, not prompt hints)

- **0% VAT**: Action layer forces vatType.id=5 when description contains befreit/exempt/fritatt/isento/0%
- **No --reload**: Server must be restarted manually after code changes
- **Payment amount**: Uses invoice outstanding balance, not prompt amount
- **userType**: Never downgrades to NO_ACCESS, generates placeholder email
- **Travel redirect**: /expense/ -> /travelExpense/ automatically
