# Accounting Family Scoreboard

Generated: `2026-03-21T19:58:27.667894+00:00`
Projection status: `noisy_calibration`

## Best Observed / Estimated By Family

- supplier: est=100.0 (observed), observed_best=100.0, projected=85.3, gap_to_100=0.0, clean=0.862, priority=n/a, blockers=none, missing=none
- customer: est=100.0 (observed), observed_best=100.0, projected=94.1, gap_to_100=0.0, clean=0.912, priority=n/a, blockers=none, missing=none
- bank_reconciliation: est=100.0 (projected), observed_best=n/a, projected=100.0, gap_to_100=0.0, clean=1.0, priority=n/a, blockers=none, missing=none
- invoice: est=100.0 (observed), observed_best=100.0, projected=64.0, gap_to_100=0.0, clean=0.743, priority=13.8, blockers=sandbox_valid_vat_type, vat_account_mapping, module_permission_blocked, missing=none
- voucher: est=85.5 (projected), observed_best=n/a, projected=85.5, gap_to_100=14.5, clean=0.864, priority=n/a, blockers=none, missing=none
- project: est=79.0 (observed), observed_best=79.0, projected=57.4, gap_to_100=21.0, clean=0.706, priority=6.0, blockers=project_hourly_rates_endpoint, module_permission_blocked, missing=project, activity, rate_type
- salary: est=70.0 (observed), observed_best=70.0, projected=92.9, gap_to_100=30.0, clean=0.905, priority=n/a, blockers=none, missing=none
- ledger_correction: est=70.0 (observed), observed_best=70.0, projected=90.0, gap_to_100=30.0, clean=0.889, priority=n/a, blockers=none, missing=none
- department: est=38.5 (projected), observed_best=n/a, projected=38.5, gap_to_100=61.5, clean=0.6, priority=10.25, blockers=duplicate_identifier, missing=department
- travel_expense: est=10.8 (projected), observed_best=n/a, projected=10.8, gap_to_100=89.2, clean=0.444, priority=12.75, blockers=employee_time_access, rate_category_date_mismatch, travel_expense_kind, missing=delivered_state, employee, rate_type
- annual_close: est=2.9 (projected), observed_best=n/a, projected=2.9, gap_to_100=97.1, clean=0.4, priority=3.0, blockers=none, missing=none
- cost_analysis: est=0.0 (projected), observed_best=n/a, projected=0.0, gap_to_100=100.0, clean=0.2, priority=3.35, blockers=activity_type_required, missing=activity, employee
- employee: est=0.0 (observed), observed_best=0.0, projected=23.6, gap_to_100=100.0, clean=0.516, priority=17.1, blockers=email_validation, missing=employee, department
- timesheet: est=0.0 (projected), observed_best=n/a, projected=0.0, gap_to_100=100.0, clean=0.286, priority=21.7, blockers=sandbox_valid_vat_type, employee_time_access, activity_type_required, missing=activity, project, employee
- product: est=0.0 (projected), observed_best=n/a, projected=0.0, gap_to_100=100.0, clean=0.36, priority=44.65, blockers=sandbox_valid_vat_type, missing=none

## Fix First

- product: priority=44.65, est=0.0, gap=100.0, blockers=sandbox_valid_vat_type
- timesheet: priority=21.70, est=0.0, gap=100.0, blockers=sandbox_valid_vat_type, employee_time_access, activity_type_required, vat_account_mapping
- employee: priority=17.10, est=0.0, gap=100.0, blockers=email_validation
- invoice: priority=13.80, est=100.0, gap=0.0, blockers=sandbox_valid_vat_type, vat_account_mapping, module_permission_blocked
- travel_expense: priority=12.75, est=10.8, gap=89.2, blockers=employee_time_access, rate_category_date_mismatch, travel_expense_kind, travel_expense_contents_required
- department: priority=10.25, est=38.5, gap=61.5, blockers=duplicate_identifier
- project: priority=6.00, est=79.0, gap=21.0, blockers=project_hourly_rates_endpoint, module_permission_blocked
- cost_analysis: priority=3.35, est=0.0, gap=100.0, blockers=activity_type_required
