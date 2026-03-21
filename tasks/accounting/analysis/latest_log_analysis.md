# Accounting Log Analysis

Analyzed `17` runs from `/tmp/accounting-logs`.

## Global

- Seen families: customer, department, employee, project, salary, supplier, timesheet, travel_expense, voucher
- New families since last run: department, timesheet
- Unseen playbook families: invoice, product
- Empty attachment runs: 0

## Alerts

- customer clean rate is 50.0%; likely missing fields: customer
- employee clean rate is 50.0%; likely missing fields: employee
- project clean rate is 0.0%; likely missing fields: project, activity, rate_type
- supplier clean rate is 50.0%; likely missing fields: none
- timesheet clean rate is 0.0%; likely missing fields: vat_type
- travel_expense clean rate is 50.0%; likely missing fields: delivered_state, rate_type, travel_expense_type_or_travel_details, cost_lines_or_allowances
- travel_expense remains partial: focus on delivered_state, rate_type, travel_expense typing, and per-diem completion before broad retries
- voucher clean rate is 50.0%; likely missing fields: none
- New families seen: department, timesheet
- Still unseen families: invoice, product

## Families

### customer

- Runs: 2
- Proxy clean rate: 50.0%
- Likely full runs: 1
- Likely partial runs: 1
- Mean API errors: 2.00
- Prompt-required fields: none
- Missing-field hypotheses: customer
- Top error: /ledger/voucher: HTTP 422 with no captured body

### department

- Runs: 1
- Proxy clean rate: 100.0%
- Likely full runs: 1
- Likely partial runs: 0
- Mean API errors: 0.00
- Prompt-required fields: none
- Missing-field hypotheses: none

### employee

- Runs: 4
- Proxy clean rate: 50.0%
- Likely full runs: 2
- Likely partial runs: 0
- Mean API errors: 0.50
- Prompt-required fields: none
- Missing-field hypotheses: employee
- Top error: /employee: HTTP 422 with no captured body

### project

- Runs: 1
- Proxy clean rate: 0.0%
- Likely full runs: 0
- Likely partial runs: 1
- Mean API errors: 10.00
- Prompt-required fields: none
- Missing-field hypotheses: project, activity, rate_type
- Top error: /timesheet/entry: HTTP 422 with no captured body

### salary

- Runs: 2
- Proxy clean rate: 100.0%
- Likely full runs: 0
- Likely partial runs: 2
- Mean API errors: 2.00
- Prompt-required fields: none
- Missing-field hypotheses: employee
- Top error: /employee/employment: HTTP 422 with no captured body

### supplier

- Runs: 2
- Proxy clean rate: 50.0%
- Likely full runs: 1
- Likely partial runs: 1
- Mean API errors: 4.00
- Prompt-required fields: none
- Missing-field hypotheses: none
- Top error: /incomingInvoice: HTTP 422 with no captured body

### timesheet

- Runs: 1
- Proxy clean rate: 0.0%
- Likely full runs: 0
- Likely partial runs: 0
- Mean API errors: 4.00
- Prompt-required fields: none
- Missing-field hypotheses: vat_type
- Top error: /invoice: HTTP 422 with no captured body
- Hypothesis: Scorer likely checks the resolved employee, project, activity, date, and exact hours, not just whether `/timesheet/entry` was posted.
- Hypothesis: Combined prompts may also require downstream invoice linkage after the hours are logged.

### travel_expense

- Runs: 2
- Proxy clean rate: 50.0%
- Likely full runs: 0
- Likely partial runs: 2
- Mean API errors: 3.00
- Prompt-required fields: per_diem, duration, departure_or_destination
- Missing-field hypotheses: delivered_state, rate_type, travel_expense_type_or_travel_details, cost_lines_or_allowances, per_diem_completion, project
- Top error: /travelExpense/:deliver: HTTP 422 with no captured body
- Hypothesis: Scorer likely checks that the travel expense is not only created, but typed correctly as travel, populated with per diem or cost details, and delivered or otherwise finalized.
- Hypothesis: Repeated validation errors around `rateType.id` suggest the per diem object is structurally incomplete even when the travel expense itself is created.

### voucher

- Runs: 2
- Proxy clean rate: 50.0%
- Likely full runs: 1
- Likely partial runs: 0
- Mean API errors: 5.00
- Prompt-required fields: none
- Missing-field hypotheses: none
- Top error: /ledger/posting: HTTP 422 with no captured body

