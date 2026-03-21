# Accounting Log Analysis

Analyzed `272` runs from `/tmp/accounting-logs`.

## Global

- Seen families: annual_close, bank_reconciliation, cost_analysis, customer, department, employee, invoice, ledger_correction, product, project, salary, supplier, timesheet, travel_expense, voucher
- New families since last run: none
- Unseen playbook families: none
- Empty attachment runs: 0

## Alerts

- annual_close clean rate is 25.0%; blockers: none; likely missing fields: none
- cost_analysis clean rate is 0.0%; blockers: activity_type_required; likely missing fields: activity, employee
- department clean rate is 60.9%; blockers: duplicate_identifier; likely missing fields: department
- employee clean rate is 48.3%; blockers: email_validation; likely missing fields: employee, department
- invoice clean rate is 66.7%; blockers: sandbox_valid_vat_type, vat_account_mapping, module_permission_blocked; likely missing fields: none
- product clean rate is 27.3%; blockers: sandbox_valid_vat_type; likely missing fields: none
- timesheet clean rate is 28.6%; blockers: sandbox_valid_vat_type, employee_time_access, activity_type_required, vat_account_mapping; likely missing fields: activity, project, employee, hours
- travel_expense clean rate is 41.2%; blockers: employee_time_access, rate_category_date_mismatch, travel_expense_kind, travel_expense_contents_required; likely missing fields: delivered_state, employee, rate_type, project
- travel_expense remains partial: focus on delivered_state, rate_type, travel_expense typing, and per-diem completion before broad retries

## Families

### annual_close

- Runs: 4
- Proxy clean rate: 25.0%
- Likely full runs: 1
- Likely partial runs: 0
- Mean API errors: 1.75
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: none
- Top error: /ledger/account: {"error":"Invalid or expired proxy token. Each submission receives a unique token - do not reuse tokens from previous submissions.","source":"nmiai-proxy"}

### bank_reconciliation

- Runs: 7
- Proxy clean rate: 100.0%
- Likely full runs: 7
- Likely partial runs: 0
- Mean API errors: 0.00
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: none

### cost_analysis

- Runs: 2
- Proxy clean rate: 0.0%
- Likely full runs: 0
- Likely partial runs: 2
- Mean API errors: 10.00
- Prompt-required fields: none
- Likely blockers: activity_type_required
- Missing-field hypotheses: activity, employee
- Top error: /activity: HTTP 422 with no captured body

### customer

- Runs: 30
- Proxy clean rate: 90.0%
- Likely full runs: 25
- Likely partial runs: 4
- Mean API errors: 0.67
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: customer, vat_type
- Top error: /ledger/voucher: HTTP 422 with no captured body

### department

- Runs: 23
- Proxy clean rate: 60.9%
- Likely full runs: 12
- Likely partial runs: 2
- Mean API errors: 0.57
- Prompt-required fields: none
- Likely blockers: duplicate_identifier
- Missing-field hypotheses: department
- Top error: /department: HTTP 422 with no captured body

### employee

- Runs: 29
- Proxy clean rate: 48.3%
- Likely full runs: 12
- Likely partial runs: 2
- Mean API errors: 0.62
- Prompt-required fields: none
- Likely blockers: email_validation
- Missing-field hypotheses: employee, department
- Top error: /employee: HTTP 422 with no captured body

### invoice

- Runs: 27
- Proxy clean rate: 66.7%
- Likely full runs: 17
- Likely partial runs: 3
- Mean API errors: 1.63
- Prompt-required fields: none
- Likely blockers: sandbox_valid_vat_type, vat_account_mapping, module_permission_blocked
- Missing-field hypotheses: none
- Top error: /invoice: HTTP 422 with no captured body

### ledger_correction

- Runs: 7
- Proxy clean rate: 85.7%
- Likely full runs: 5
- Likely partial runs: 2
- Mean API errors: 1.14
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: none
- Top error: /ledger/voucher: HTTP 422 with no captured body

### product

- Runs: 22
- Proxy clean rate: 27.3%
- Likely full runs: 6
- Likely partial runs: 2
- Mean API errors: 16.00
- Prompt-required fields: none
- Likely blockers: sandbox_valid_vat_type
- Missing-field hypotheses: none
- Top error: /product: HTTP 422 with no captured body

### project

- Runs: 16
- Proxy clean rate: 75.0%
- Likely full runs: 5
- Likely partial runs: 9
- Mean API errors: 3.38
- Prompt-required fields: none
- Likely blockers: project_hourly_rates_endpoint, module_permission_blocked
- Missing-field hypotheses: project, activity, rate_type
- Top error: /project/hourlyRates/updateOrAddHourRates: HTTP 422 with no captured body

### salary

- Runs: 18
- Proxy clean rate: 88.9%
- Likely full runs: 14
- Likely partial runs: 2
- Mean API errors: 0.22
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: employee
- Top error: /employee/employment: HTTP 422 with no captured body

### supplier

- Runs: 27
- Proxy clean rate: 88.9%
- Likely full runs: 24
- Likely partial runs: 2
- Mean API errors: 0.81
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: project, activity
- Top error: /timesheet/entry: dateFrom -> Kan ikke være null.

### timesheet

- Runs: 21
- Proxy clean rate: 28.6%
- Likely full runs: 4
- Likely partial runs: 5
- Mean API errors: 2.67
- Prompt-required fields: project, activity
- Likely blockers: sandbox_valid_vat_type, employee_time_access, activity_type_required, vat_account_mapping, project_hourly_rates_endpoint, duplicate_identifier
- Missing-field hypotheses: activity, project, employee, hours, department
- Top error: /timesheet/entry: HTTP 422 with no captured body
- Hypothesis: Scorer likely checks the resolved employee, project, activity, date, and exact hours, not just whether `/timesheet/entry` was posted.
- Hypothesis: Combined prompts may also require downstream invoice linkage after the hours are logged.

### travel_expense

- Runs: 17
- Proxy clean rate: 41.2%
- Likely full runs: 5
- Likely partial runs: 4
- Mean API errors: 2.18
- Prompt-required fields: duration, departure_or_destination, per_diem
- Likely blockers: employee_time_access, rate_category_date_mismatch, travel_expense_kind, travel_expense_contents_required, per_diem_rate_required
- Missing-field hypotheses: delivered_state, employee, rate_type, project, travel_expense_type_or_travel_details, cost_lines_or_allowances
- Top error: /travelExpense: HTTP 422 with no captured body
- Hypothesis: Scorer likely checks that the travel expense is not only created, but typed correctly as travel, populated with per diem or cost details, and delivered or otherwise finalized.
- Hypothesis: Repeated validation errors around `rateType.id` suggest the per diem object is structurally incomplete even when the travel expense itself is created.

### voucher

- Runs: 22
- Proxy clean rate: 86.4%
- Likely full runs: 17
- Likely partial runs: 3
- Mean API errors: 0.91
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: none
- Top error: /ledger/posting: HTTP 422 with no captured body

