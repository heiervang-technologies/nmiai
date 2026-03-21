# Accounting Log Analysis

Analyzed `137` runs from `/tmp/accounting-logs`.

## Global

- Seen families: customer, department, employee, invoice, product, project, salary, supplier, timesheet, travel_expense, voucher
- New families since last run: none
- Unseen playbook families: bank_reconciliation, ledger_correction
- Empty attachment runs: 0

## Alerts

- department clean rate is 27.3%; blockers: duplicate_identifier; likely missing fields: none
- employee clean rate is 25.0%; blockers: email_validation; likely missing fields: employee, department
- invoice clean rate is 12.5%; blockers: sandbox_valid_vat_type, vat_account_mapping; likely missing fields: none
- product clean rate is 6.7%; blockers: sandbox_valid_vat_type; likely missing fields: none
- timesheet clean rate is 7.1%; blockers: sandbox_valid_vat_type, employee_time_access, activity_type_required, vat_account_mapping; likely missing fields: activity, project, employee, hours
- travel_expense clean rate is 10.0%; blockers: employee_time_access, travel_expense_kind, travel_expense_contents_required, per_diem_rate_required; likely missing fields: delivered_state, rate_type, employee, project
- travel_expense remains partial: focus on delivered_state, rate_type, travel_expense typing, and per-diem completion before broad retries
- Still unseen families: bank_reconciliation, ledger_correction

## Families

### customer

- Runs: 12
- Proxy clean rate: 83.3%
- Likely full runs: 10
- Likely partial runs: 1
- Mean API errors: 1.17
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: customer
- Top error: /ledger/voucher: HTTP 422 with no captured body

### department

- Runs: 11
- Proxy clean rate: 27.3%
- Likely full runs: 2
- Likely partial runs: 1
- Mean API errors: 1.09
- Prompt-required fields: none
- Likely blockers: duplicate_identifier
- Missing-field hypotheses: none
- Top error: /department: HTTP 422 with no captured body

### employee

- Runs: 20
- Proxy clean rate: 25.0%
- Likely full runs: 4
- Likely partial runs: 1
- Mean API errors: 0.80
- Prompt-required fields: none
- Likely blockers: email_validation
- Missing-field hypotheses: employee, department
- Top error: /employee: HTTP 422 with no captured body

### invoice

- Runs: 8
- Proxy clean rate: 12.5%
- Likely full runs: 1
- Likely partial runs: 1
- Mean API errors: 3.75
- Prompt-required fields: none
- Likely blockers: sandbox_valid_vat_type, vat_account_mapping
- Missing-field hypotheses: none
- Top error: /invoice: HTTP 422 with no captured body

### product

- Runs: 15
- Proxy clean rate: 6.7%
- Likely full runs: 1
- Likely partial runs: 2
- Mean API errors: 23.47
- Prompt-required fields: none
- Likely blockers: sandbox_valid_vat_type
- Missing-field hypotheses: none
- Top error: /product: HTTP 422 with no captured body

### project

- Runs: 8
- Proxy clean rate: 75.0%
- Likely full runs: 0
- Likely partial runs: 7
- Mean API errors: 3.00
- Prompt-required fields: none
- Likely blockers: project_hourly_rates_endpoint, module_permission_blocked
- Missing-field hypotheses: project, activity, rate_type
- Top error: /project/hourlyRates/updateOrAddHourRates: HTTP 422 with no captured body

### salary

- Runs: 12
- Proxy clean rate: 100.0%
- Likely full runs: 10
- Likely partial runs: 2
- Mean API errors: 0.33
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: employee
- Top error: /employee/employment: HTTP 422 with no captured body

### supplier

- Runs: 9
- Proxy clean rate: 88.9%
- Likely full runs: 8
- Likely partial runs: 1
- Mean API errors: 0.89
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: none
- Top error: /incomingInvoice: HTTP 422 with no captured body

### timesheet

- Runs: 14
- Proxy clean rate: 7.1%
- Likely full runs: 1
- Likely partial runs: 3
- Mean API errors: 3.14
- Prompt-required fields: project, activity
- Likely blockers: sandbox_valid_vat_type, employee_time_access, activity_type_required, vat_account_mapping, project_hourly_rates_endpoint
- Missing-field hypotheses: activity, project, employee, hours
- Top error: /timesheet/entry: HTTP 422 with no captured body
- Hypothesis: Scorer likely checks the resolved employee, project, activity, date, and exact hours, not just whether `/timesheet/entry` was posted.
- Hypothesis: Combined prompts may also require downstream invoice linkage after the hours are logged.

### travel_expense

- Runs: 10
- Proxy clean rate: 10.0%
- Likely full runs: 0
- Likely partial runs: 3
- Mean API errors: 3.00
- Prompt-required fields: per_diem, duration, departure_or_destination
- Likely blockers: employee_time_access, travel_expense_kind, travel_expense_contents_required, per_diem_rate_required, rate_category_date_mismatch
- Missing-field hypotheses: delivered_state, rate_type, employee, project, travel_expense_type_or_travel_details, cost_lines_or_allowances
- Top error: /travelExpense: HTTP 422 with no captured body
- Hypothesis: Scorer likely checks that the travel expense is not only created, but typed correctly as travel, populated with per diem or cost details, and delivered or otherwise finalized.
- Hypothesis: Repeated validation errors around `rateType.id` suggest the per diem object is structurally incomplete even when the travel expense itself is created.

### voucher

- Runs: 18
- Proxy clean rate: 88.9%
- Likely full runs: 15
- Likely partial runs: 2
- Mean API errors: 0.89
- Prompt-required fields: none
- Likely blockers: none
- Missing-field hypotheses: none
- Top error: /ledger/posting: HTTP 422 with no captured body

