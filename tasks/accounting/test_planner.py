#!/usr/bin/env python3
"""
Test the planner's keyword classifier — ZERO LLM calls.
Only tests classify_by_keywords, not the LLM fallback.
Validates that prompts route to the correct task family.

Usage: cd tasks/accounting/server && uv run python ../test_planner.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "server"))
from planner import classify_by_keywords

# (prompt, expected_family)
CASES = [
    # Norwegian
    ("Opprett en ansatt med navn Kari Nordmann", "employee"),
    ("Opprett en kunde med navn Firma AS", "customer"),
    ("Opprett avdeling Salg med nummer 200", "department"),
    ("Opprett et produkt med pris 1500 kr", "product"),
    ("Opprett en faktura til kunde Firma AS", "invoice"),
    ("Registrer betaling på faktura 1001", "invoice"),
    ("Opprett en kreditnota for faktura 1001", "invoice"),
    ("Registrer en reiseregning for ansatt", "travel_expense"),
    ("Registrer 8 timer på prosjekt X", "timesheet"),
    ("Kjør lønn for ansatt nummer 1, grunnlønn 45000", "salary"),
    ("Opprett prosjekt Ny Nettside for kunde X", "project"),
    ("Opprett et bilag: debet konto 6300", "voucher"),
    ("Registrer en leverandørfaktura fra Firma AS", "supplier"),
    # English
    ("Create an employee named John Smith", "employee"),
    ("Create a customer called Acme Corp", "customer"),
    ("Register a travel expense for employee 1", "travel_expense"),
    ("Log 8 hours on project Alpha", "timesheet"),
    ("Process salary for employee, base 50000", "salary"),
    ("Create a credit note for invoice 1001", "invoice"),
    ("Register a supplier invoice from Vendor AS", "supplier"),
    # Multi-language
    ("Erstellen Sie einen Mitarbeiter Max Müller", "employee"),
    ("Créez un client nommé Société SA", "customer"),
    ("Registre una factura del proveedor ABC", "supplier"),
    # New families
    ("Vi har oppdaget feil i hovedboken", "ledger_correction"),
    ("Errors in the ledger for January", "ledger_correction"),
    ("Utfør årsavslutning med avskrivning", "annual_close"),
    ("Annual closing with depreciation", "annual_close"),
    ("Avstem bankkontoen med vedlagt CSV", "bank_reconciliation"),
    ("Reconcile bank statement with open invoices", "bank_reconciliation"),
    ("Total costs increased significantly, analyze the general ledger", "cost_analysis"),
    ("Kostnadene økte, analyser hovedboken", "cost_analysis"),
    # Tricky cases — should NOT misclassify
    ("Opprett en faktura for 10 konsulenttjenester à 1500 kr", "invoice"),  # not timesheet
    ("Opprett et bilag for purregebyr på kunde X", "voucher"),  # not customer
]

passed = 0
failed = 0
for prompt, expected in CASES:
    family, confidence = classify_by_keywords(prompt)
    ok = family == expected
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: '{prompt[:60]}...' -> got '{family}' (expected '{expected}', conf={confidence})")

total = passed + failed
print(f"\nPLANNER CLASSIFICATION: {passed}/{total} ({passed/total*100:.0f}%)")
if failed == 0:
    print("All classifications correct!")
