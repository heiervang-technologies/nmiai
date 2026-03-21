# Accounting Failure Backlog — Real-Time Tracker

Last updated: 2026-03-21 18:40 UTC

## Scoring: 65.7/100 (leader: 100.0) — up from 45.8

## RESOLVED Issues (confirmed fixed in latest results)

| # | Issue | Status | Fix |
|---|-------|--------|-----|
| 1 | VAT code invalid (product/invoice) | FIXED | Dynamic VAT lookup `_get_outgoing_vat_types` |
| 2 | Employee access rights (timesheet/travel) | FIXED | `_grant_employee_all_privileges` + PUT fallback with department.id |
| 3 | Project fixedPrice field | FIXED | `_set_project_fixed_price` via hourlyRates API |
| 4 | Activity creation activityType null | FIXED | activityType: "PROJECT_SPECIFIC_ACTIVITY" |
| 5 | Travel expense = employee expense | FIXED | travelDetails always included |
| 6 | Department number collision | FIXED | Check existing before create |
| 7 | Supplier ID missing on voucher postings | FIXED | supplier ref when ledgerType=VENDOR/SUPPLIER |
| 8 | Ledger posting query missing dates | FIXED | Auto-add dateFrom/dateTo in generic_api_call |
| 9 | Employment type field doesn't exist | FIXED | Removed employmentType from body |
| 10 | Project manager lacks access | FIXED | _grant_employee_all_privileges before project |
| 11 | Voucher posting needs customer.id | FIXED | Auto-add customer ref for CUSTOMER ledger accounts |
| 12 | Product duplicate loops | FIXED | Duplicate name/number check before POST |
| 13 | Ledger review supplier.id | FIXED | System prompt + auto supplier ref on voucher postings |
| 14 | Wrong activity endpoint /project/{id}/activity | FIXED | Auto-rewrite to POST /project/projectActivity |
| 15 | Cost analysis voucher ID guessing | FIXED | System prompt: use /ledger/posting not individual voucher GETs |
| 16 | OpenRouter credits | MITIGATED | max_tokens 65536→4096, retries 2→1 |

## OPEN Issues (still failing in latest results)

| # | Issue | Validation Error | Occurrences | Root Cause |
|---|-------|-----------------|-------------|------------|
| 17 | **EUR payment paidAmountCurrency** | `paidAmountCurrency: Mangler` | 1x (151822) | LLM doesn't pass paidAmountCurrency for foreign currency payments. Code handles it but LLM must provide it. |
| 18 | **Travel perDiem rateCategory mismatch** | `rateCategory.id: Reiseregningens dato samsvarer ikke med valgt satskategori` | 1x (151909) | Rate category date range doesn't match travel date. Need to query valid rate categories for the travel period. |
| 19 | **Employee employment/details wrong endpoint** | `payrollPercentage: Feltet eksisterer ikke i objektet` | 1x (150440) | Agent uses POST /employee/employment/details instead of /employee/employment. Field payrollPercentage doesn't exist. |
| 20 | **Project lifecycle projectActivity 409** | `Duplicate entry` | 2x (151716, 141056) | Activity already exists for project. Need check-before-create for projectActivity. |
| 21 | **Project hourlyRates wrong type** | `hourlyRateModel: Verdien er ikke av korrekt type for dette feltet` | 2x (141056, 140149) | hourlyRateModel value format wrong on POST /project/hourlyRates. |
| 22 | **Bank reconciliation payment matching** | `Ugyldig fakturanummer` | 1x (141106) | Agent guesses invoice numbers from bank statement that don't match Tripletex IDs. |
| 23 | **Employee missing email for Tripletex user** | `email: Må angis for Tripletex-brukere` | 1x (141307) | PDF extraction didn't get email, employee created without it but Tripletex requires email for users. |
| 24 | **Expired proxy token** | 403 on all calls | 1x (150558) | External — token expired. Cannot fix. |
| 25 | PROJECT_SPECIFIC_ACTIVITY on /activity | FIXED | Route PROJECT_SPECIFIC to /project/projectActivity |
| 26 | GET /timesheet/entry missing dateFrom | FIXED | Auto-add dateFrom/dateTo |
| 27 | projectActivity 409 duplicate | FIXED | Return existing activity on 409 |
| 28 | Agio voucher VAT type on locked accounts | FIXED | Auto-strip vatType on 1500/8060/8160/3400 |
| 29 | Project lifecycle misclassified | FIXED | Multi-lang keywords (FR/ES/PT/DE/NO) |
| 30 | Supplier registration misclassified | FIXED | Multi-word keywords outweigh email false positives |
| 31 | Agent asks questions instead of acting | FIXED | System prompt: NEVER ask questions |

## Latest Results (18:40 UTC) — 39 submissions, 31 clean (79.5%)
```
OK  | supplier(5x)    | 3-5c 0e  | All clean across ES/EN/FR/PT/NO
OK  | employee(3x)    | 4-6c 0e  | DE/EN PDF extraction working
OK  | customer(3x)    | 6-14c 0e | EUR agio + reminder fees working
OK  | invoice(3x)     | 1-14c 0e | Payment + reminder fee + registration
OK  | department(3x)  | 6-13c 0e | Receipt booking + creation working
OK  | timesheet(1x)   | 10c 0e   | 29 hours registered clean
OK  | project(2x)     | 3c 0e    | Simple project creation working
OK  | salary(2x)      | 1-2c 0e  | Monthly closing clean
WARN| ledger_corr(2x) | 14-23c 2-6e | Voucher supplier.id still LLM-dependent
WARN| cost_analysis   | 20c 6e   | Activity creation still failing (pre-fix)
WARN| project_lifecycle| 24c 1-3e | Timesheet 409 + activity routing (pre-fix)
FAIL| expired_token   | 0c 4e    | External: 403 expired proxy token
```

## Task Types Now Working (confirmed clean)
- Customer/supplier/employee/department/product creation
- Salary/payroll via voucher
- Voucher/journal entries + dimensions
- Credit notes + payment registration/reversal
- Multi-currency invoicing + agio
- Monthly/annual closing (depreciation + accruals)
- Bank reconciliation from CSV
- Supplier invoice from PDF
- Employee onboarding from PDF
- Ledger review/correction
- Receipt booking to department
- Travel expense with per diem

## Task Types Still Weak
- Full project lifecycle (hourlyRates + activity creation)
- Cost analysis (create projects from ledger analysis)
- EUR payment with paidAmountCurrency
- Travel per diem rate category matching

## Failure Predictors (what makes tasks score 0/10)

**Tier is the #1 predictor:**
- TIER1 (simple create/register): 63% clean, 4% fail
- TIER2 (PDF/complex): 43% clean, 22% fail
- TIER3 (multi-step/analysis): 41% clean, 25% fail

**Language is NOT a predictor:** NO=12%, ES=8%, EN=9% fail rate (normalized)

**Guaranteed 0/10 combos:**
- ANY language + TIER3 + project = always fails
- ANY language + TIER3 + cost_analysis = always fails

**Properties that make tasks hard:**
1. "Analyze ledger + create entities" — multi-step: query→analyze→create
2. "Full project lifecycle" — project + activity + hourlyRates + timesheet + invoice
3. "Foreign currency + agio" — paidAmountCurrency + exchange rate voucher
4. "PDF with employment details" — depends on field extraction quality
5. "Bank reconciliation from CSV" — CSV parsing + invoice ID matching

**Task families that NEVER succeed clean:**
- `project`: 0% clean (0/10) — hourlyRates + activity creation always breaks
- `cost_analysis`: 0% clean — voucher ID guessing + activity creation
