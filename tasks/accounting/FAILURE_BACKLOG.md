# Accounting Failure Backlog — Real-Time Tracker

Last updated: 2026-03-21 15:05 UTC

## Scoring: 45.8/100 (leader: 100.0)

## Issue Status

| # | Issue | Errors | Status | Fix |
|---|-------|--------|--------|-----|
| 1 | **VAT code invalid** (product/invoice) | 346 | FIXED | Dynamic VAT lookup via `_get_outgoing_vat_types` |
| 2 | **Employee access rights** (timesheet/travel) | 30 | FIXED | `_grant_employee_all_privileges` + fallback PUT allowInformationRegistration |
| 3 | **Project fixedPrice field** | 6 | FIXED | `_set_project_fixed_price` via hourlyRates API |
| 4 | **Activity creation activityType null** | 18 | FIXED | activityType: "PROJECT_SPECIFIC_ACTIVITY" set |
| 5 | **Travel expense = employee expense** | 4 | FIXED | travelDetails always included |
| 6 | **Employee email validation** (john@test.com) | 14 | WONTFIX | Tripletex rejects test emails, only affects Bayesian tests |
| 7 | **Department number collision** | 12 | FIXED | Check existing before create |
| 8 | **Supplier ID missing on voucher postings** | 14 | FIXED | supplier ref added when ledgerType=VENDOR/SUPPLIER |
| 9 | **Project hourlyRates model format** | 10 | FIXED | hourlyRateModel: "TYPE_FIXED_HOURLY_RATE" |
| 10 | **Ledger posting query missing dates** | 10 | FIXED | Auto-add dateFrom/dateTo in generic_api_call |
| 11 | **Employment type field doesn't exist** | 4 | FIXED | Removed employmentType from body |
| 12 | **Project manager lacks access** | 2 | FIXED | _grant_employee_all_privileges before project creation |
| 13 | **Voucher posting needs customer.id** | 4 | FIXED | Auto-add customer ref for CUSTOMER ledger accounts |
| 14 | **OpenRouter out of credits** | 0 | MITIGATED | max_tokens 65536→4096, retries 2→1 |
| 15 | **Product duplicate 103-error loops** | 306 | FIXED | Duplicate name/number check before POST |
| 16 | **Ledger review correction vouchers** | 6 | PARTIAL | System prompt guidance added, needs supplierName in args |

## Real-Time Submission Results

Format: `STATUS | family | calls errors time | prompt_preview`

### Latest Batch (post all fixes)
```
OK  | department      | 10c 0e  12.6s | Receipt posted to department
OK  | bank_reconciliation | 23c 0e 19.5s | Bank statement CSV reconciliation
OK  | invoice         | 10c 0e  31.0s | Annual closing 2025
OK  | voucher         | 5c 0e  19.1s | Custom dimension + voucher
OK  | salary          | 3c 0e  13.2s | Salary Nynorsk
OK  | travel_expense  | 6c 0e   6.1s | Travel expense with per diem
OK  | product         | 1c 0e   6.5s | Product (duplicate check)
OK  | department      | 1c 0e   5.1s | Department (collision check)
WARN| employee        | 7c 2e  23.8s | Employee from PDF - employment creation 422
WARN| ledger_correction| 12c 2e 22.3s | Portuguese ledger review - supplier.id on correction voucher
WARN| timesheet       | 6c 2e   5.5s | Timesheet - 409 conflict (Bayesian repeat test)
```

## Task Types We Handle Well (>80% clean)
- Customer creation
- Employee creation
- Department creation
- Product creation
- Salary/payroll (voucher approach)
- Voucher/journal entries
- Accounting dimensions
- Credit notes
- Payment registration/reversal
- Supplier registration
- Multi-currency invoicing
- Monthly/annual closing
- Bank reconciliation from CSV

## Task Types Still Weak (<80% clean)
- Employee onboarding from PDF (employment creation 422)
- Full project lifecycle (hourlyRates + activity + invoice)
- Ledger review/correction (supplier.id on correction vouchers)
- Complex travel expenses with multiple cost lines
