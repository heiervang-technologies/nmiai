# Accounting Family Scoreboard

Generated: `2026-03-21T20:45:01.715442+00:00`
Projection status: `noisy_calibration`

## Best Observed / Estimated By Family

- supplier: est=100.0 (observed), observed_best=100.0, projected=84.6, gap_to_100=0.0, clean=0.862, priority=n/a, opportunity=n/a, blockers=none, missing=none
- customer: est=100.0 (observed), observed_best=100.0, projected=88.8, gap_to_100=0.0, clean=0.914, priority=n/a, opportunity=n/a, blockers=none, missing=none
- annual_close: est=100.0 (observed), observed_best=100.0, projected=60.9, gap_to_100=0.0, clean=0.571, priority=3.0, opportunity=0.0, blockers=none, missing=none
- invoice: est=100.0 (observed), observed_best=100.0, projected=77.5, gap_to_100=0.0, clean=0.775, priority=13.8, opportunity=0.0, blockers=sandbox_valid_vat_type, vat_account_mapping, module_permission_blocked, missing=none
- employee: est=100.0 (observed), observed_best=100.0, projected=57.7, gap_to_100=0.0, clean=0.531, priority=17.35, opportunity=0.0, blockers=email_validation, missing=employee, department
- product: est=100.0 (observed), observed_best=100.0, projected=47.6, gap_to_100=0.0, clean=0.407, priority=44.65, opportunity=0.0, blockers=sandbox_valid_vat_type, missing=none
- bank_reconciliation: est=95.8 (projected), observed_best=n/a, projected=95.8, gap_to_100=4.2, clean=1.0, priority=n/a, opportunity=n/a, blockers=none, missing=none
- voucher: est=84.7 (projected), observed_best=n/a, projected=84.7, gap_to_100=15.3, clean=0.864, priority=n/a, opportunity=n/a, blockers=none, missing=none
- project: est=79.0 (observed), observed_best=79.0, projected=71.9, gap_to_100=21.0, clean=0.706, priority=7.55, opportunity=1.585, blockers=project_hourly_rates_endpoint, module_permission_blocked, duplicate_identifier, missing=project, activity, rate_type
- salary: est=70.0 (observed), observed_best=70.0, projected=88.0, gap_to_100=30.0, clean=0.905, priority=n/a, opportunity=n/a, blockers=none, missing=none
- ledger_correction: est=70.0 (observed), observed_best=70.0, projected=86.7, gap_to_100=30.0, clean=0.889, priority=n/a, opportunity=n/a, blockers=none, missing=none
- department: est=63.3 (projected), observed_best=n/a, projected=63.3, gap_to_100=36.7, clean=0.6, priority=11.4, opportunity=4.184, blockers=duplicate_identifier, module_permission_blocked, missing=department
- travel_expense: est=50.6 (projected), observed_best=n/a, projected=50.6, gap_to_100=49.4, clean=0.444, priority=12.8, opportunity=6.323, blockers=employee_time_access, rate_category_date_mismatch, travel_expense_kind, missing=delivered_state, employee, rate_type
- timesheet: est=37.7 (projected), observed_best=n/a, projected=37.7, gap_to_100=62.3, clean=0.286, priority=21.7, opportunity=13.519, blockers=sandbox_valid_vat_type, employee_time_access, activity_type_required, missing=activity, project, employee
- cost_analysis: est=28.1 (projected), observed_best=n/a, projected=28.1, gap_to_100=71.9, clean=0.167, priority=6.55, opportunity=4.709, blockers=activity_type_required, missing=activity, employee, project

## Fix First

- timesheet: opportunity=13.519, priority=21.70, est=37.7, gap=62.3, blockers=sandbox_valid_vat_type, employee_time_access, activity_type_required, vat_account_mapping
- travel_expense: opportunity=6.323, priority=12.80, est=50.6, gap=49.4, blockers=employee_time_access, rate_category_date_mismatch, travel_expense_kind, travel_expense_contents_required
- cost_analysis: opportunity=4.709, priority=6.55, est=28.1, gap=71.9, blockers=activity_type_required
- department: opportunity=4.184, priority=11.40, est=63.3, gap=36.7, blockers=duplicate_identifier, module_permission_blocked
- project: opportunity=1.585, priority=7.55, est=79.0, gap=21.0, blockers=project_hourly_rates_endpoint, module_permission_blocked, duplicate_identifier
- product: opportunity=0.0, priority=44.65, est=100.0, gap=0.0, blockers=sandbox_valid_vat_type
- employee: opportunity=0.0, priority=17.35, est=100.0, gap=0.0, blockers=email_validation
- invoice: opportunity=0.0, priority=13.80, est=100.0, gap=0.0, blockers=sandbox_valid_vat_type, vat_account_mapping, module_permission_blocked
