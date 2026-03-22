# Accounting Family Scoreboard

Generated: `2026-03-22T06:41:05.937777+00:00`
Projection status: `noisy_calibration`

## Best Observed / Estimated By Family

- bank_reconciliation: est=95.8 (projected), observed_best=n/a, projected=95.8, gap_to_100=4.2, clean=1.0, priority=n/a, opportunity=n/a, blockers=none, missing=none
- customer: est=94.4 (stabilized), observed_best=100.0, projected=88.8, gap_to_100=5.6, clean=0.914, priority=n/a, opportunity=n/a, blockers=none, missing=none
- supplier: est=89.7 (stabilized), observed_best=100.0, projected=84.6, gap_to_100=10.3, clean=0.862, priority=n/a, opportunity=n/a, blockers=none, missing=none
- voucher: est=84.7 (projected), observed_best=n/a, projected=84.7, gap_to_100=15.3, clean=0.864, priority=n/a, opportunity=n/a, blockers=none, missing=none
- salary: est=82.0 (stabilized), observed_best=70.0, projected=88.0, gap_to_100=18.0, clean=0.905, priority=n/a, opportunity=n/a, blockers=none, missing=none
- ledger_correction: est=81.2 (stabilized), observed_best=70.0, projected=86.7, gap_to_100=18.8, clean=0.889, priority=n/a, opportunity=n/a, blockers=none, missing=none
- invoice: est=79.1 (stabilized), observed_best=100.0, projected=77.5, gap_to_100=20.9, clean=0.775, priority=13.8, opportunity=2.884, blockers=sandbox_valid_vat_type, vat_account_mapping, module_permission_blocked, missing=none
- project: est=74.2 (stabilized), observed_best=79.0, projected=71.9, gap_to_100=25.8, clean=0.706, priority=7.55, opportunity=1.948, blockers=project_hourly_rates_endpoint, module_permission_blocked, duplicate_identifier, missing=project, activity, rate_type
- product: est=65.1 (stabilized), observed_best=100.0, projected=47.6, gap_to_100=34.9, clean=0.407, priority=44.65, opportunity=15.583, blockers=sandbox_valid_vat_type, missing=none
- department: est=63.3 (projected), observed_best=n/a, projected=63.3, gap_to_100=36.7, clean=0.6, priority=11.4, opportunity=4.184, blockers=duplicate_identifier, module_permission_blocked, missing=department
- employee: est=55.1 (stabilized), observed_best=100.0, projected=57.7, gap_to_100=44.9, clean=0.531, priority=17.35, opportunity=7.79, blockers=email_validation, missing=employee, department
- annual_close: est=51.0 (stabilized), observed_best=100.0, projected=60.9, gap_to_100=49.0, clean=0.571, priority=3.0, opportunity=1.47, blockers=none, missing=none
- travel_expense: est=50.6 (projected), observed_best=n/a, projected=50.6, gap_to_100=49.4, clean=0.444, priority=12.8, opportunity=6.323, blockers=employee_time_access, rate_category_date_mismatch, travel_expense_kind, missing=delivered_state, employee, rate_type
- timesheet: est=37.7 (projected), observed_best=n/a, projected=37.7, gap_to_100=62.3, clean=0.286, priority=21.7, opportunity=13.519, blockers=sandbox_valid_vat_type, employee_time_access, activity_type_required, missing=activity, project, employee
- cost_analysis: est=28.1 (projected), observed_best=n/a, projected=28.1, gap_to_100=71.9, clean=0.167, priority=6.55, opportunity=4.709, blockers=activity_type_required, missing=activity, employee, project

## Fix First

- product: opportunity=15.583, priority=44.65, est=65.1, gap=34.9, blockers=sandbox_valid_vat_type
- timesheet: opportunity=13.519, priority=21.70, est=37.7, gap=62.3, blockers=sandbox_valid_vat_type, employee_time_access, activity_type_required, vat_account_mapping
- employee: opportunity=7.79, priority=17.35, est=55.1, gap=44.9, blockers=email_validation
- travel_expense: opportunity=6.323, priority=12.80, est=50.6, gap=49.4, blockers=employee_time_access, rate_category_date_mismatch, travel_expense_kind, travel_expense_contents_required
- cost_analysis: opportunity=4.709, priority=6.55, est=28.1, gap=71.9, blockers=activity_type_required
- department: opportunity=4.184, priority=11.40, est=63.3, gap=36.7, blockers=duplicate_identifier, module_permission_blocked
- invoice: opportunity=2.884, priority=13.80, est=79.1, gap=20.9, blockers=sandbox_valid_vat_type, vat_account_mapping, module_permission_blocked
- project: opportunity=1.948, priority=7.55, est=74.2, gap=25.8, blockers=project_hourly_rates_endpoint, module_permission_blocked, duplicate_identifier
