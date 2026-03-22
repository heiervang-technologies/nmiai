"""
Adversarial prompts for testing weak families against known failure modes.
Each prompt has extractable ground truth for scorer validation.
Languages: nb (Norwegian Bokmål), nn (Nynorsk), en, de, fr, es, pt
"""

ADVERSARIAL_PROMPTS = [
    # ════════════════════════════════════════════════════════════════
    #  TRAVEL EXPENSE  (scoring 38%)
    # ════════════════════════════════════════════════════════════════

    # 1. Foreign travel with per diem (tests isForeignTravel + rateCategory)
    {
        "family": "travel_expense",
        "language": "nb",
        "difficulty": "hard",
        "prompt": (
            "Registrer en reiseregning for Kari Nilsen (kari.nilsen@example.org) "
            "for «Kundemøte i Berlin». Reisen varte fra 10. mars til 13. mars 2026 (3 dager). "
            "Utenlandsreise med diett, dagsats 800 kr per dag. "
            "Hun overnattet på hotell (dekket av arbeidsgiver). "
            "Legg ved utlegg: flybillett 4200 kr og drosje 350 kr. "
            "Lever reiseregningen etter registrering."
        ),
        "expected_fields": {
            "employee_email": "kari.nilsen@example.org",
            "title": "Kundemøte i Berlin",
            "isForeignTravel": True,
            "perDiem_days": 3,
            "perDiem_rate": 800,
            "overnightAccommodation": "HOTEL",
            "costs": [
                {"description": "Flybillett", "amount": 4200},
                {"description": "Drosje", "amount": 350},
            ],
            "delivered": True,
        },
        "failure_mode_tested": "foreign_travel_per_diem_hotel",
    },

    # 2. Domestic travel, no per diem, only costs (tests costCategory + deliver)
    {
        "family": "travel_expense",
        "language": "en",
        "difficulty": "medium",
        "prompt": (
            "Create a travel expense report for Thomas Berg (thomas.berg@example.org) "
            "titled \"Client visit Trondheim\". Domestic trip, 2 days (March 15-16, 2026). "
            "Costs: train ticket 1850 NOK, hotel 1400 NOK, parking 250 NOK. "
            "All paid by employee. Submit the expense when done."
        ),
        "expected_fields": {
            "employee_email": "thomas.berg@example.org",
            "title": "Client visit Trondheim",
            "isForeignTravel": False,
            "costs": [
                {"description": "Train ticket", "amount": 1850},
                {"description": "Hotel", "amount": 1400},
                {"description": "Parking", "amount": 250},
            ],
            "delivered": True,
        },
        "failure_mode_tested": "domestic_multi_cost_deliver",
    },

    # 3. French - foreign travel with per diem + accommodation
    {
        "family": "travel_expense",
        "language": "fr",
        "difficulty": "hard",
        "prompt": (
            "Enregistrez une note de frais pour Pierre Dupont (pierre.dupont@example.org) "
            "intitulée «Visite client Stockholm». Voyage à l'étranger du 5 au 8 mars 2026 "
            "(3 jours). Per diem de 900 NOK par jour, hébergement à l'hôtel (payé par l'employeur). "
            "Frais supplémentaires : taxi 600 NOK. "
            "Soumettez la note de frais après l'enregistrement."
        ),
        "expected_fields": {
            "employee_email": "pierre.dupont@example.org",
            "title": "Visite client Stockholm",
            "isForeignTravel": True,
            "perDiem_days": 3,
            "perDiem_rate": 900,
            "overnightAccommodation": "HOTEL",
            "costs": [{"description": "Taxi", "amount": 600}],
            "delivered": True,
        },
        "failure_mode_tested": "foreign_travel_french_per_diem",
    },

    # 4. German - mileage allowance (tests different cost type)
    {
        "family": "travel_expense",
        "language": "de",
        "difficulty": "medium",
        "prompt": (
            "Erfassen Sie eine Reisekostenabrechnung für Max Fischer (max.fischer@example.org) "
            "mit dem Titel «Kundenbesuch Stavanger». Inlandsreise, 1 Tag (18. März 2026). "
            "Kilometergeld: 340 km × 3,50 NOK = 1190 NOK. "
            "Parkgebühren: 200 NOK. Reichen Sie die Abrechnung ein."
        ),
        "expected_fields": {
            "employee_email": "max.fischer@example.org",
            "title": "Kundenbesuch Stavanger",
            "isForeignTravel": False,
            "costs": [
                {"description": "Kilometergeld", "amount": 1190},
                {"description": "Parkgebühren", "amount": 200},
            ],
            "delivered": True,
        },
        "failure_mode_tested": "mileage_allowance_german",
    },

    # 5. Nynorsk - edge case: per diem with private accommodation
    {
        "family": "travel_expense",
        "language": "nn",
        "difficulty": "hard",
        "prompt": (
            "Registrer ei reiserekning for Astrid Haugen (astrid.haugen@example.org) "
            "for «Konferanse i København». Utanlandsreise frå 20. til 23. mars 2026 (3 dagar). "
            "Diett: 750 kr per dag. Ho budde privat (ikkje hotell). "
            "Utlegg: flybillett 3800 kr, konferanseavgift 2500 kr. "
            "Lever reiserekninga."
        ),
        "expected_fields": {
            "employee_email": "astrid.haugen@example.org",
            "title": "Konferanse i København",
            "isForeignTravel": True,
            "perDiem_days": 3,
            "perDiem_rate": 750,
            "overnightAccommodation": "NONE",
            "costs": [
                {"description": "Flybillett", "amount": 3800},
                {"description": "Konferanseavgift", "amount": 2500},
            ],
            "delivered": True,
        },
        "failure_mode_tested": "foreign_nynorsk_private_accommodation",
    },

    # 6. Spanish - simple domestic
    {
        "family": "travel_expense",
        "language": "es",
        "difficulty": "easy",
        "prompt": (
            "Registre un informe de gastos de viaje para Elena García (elena.garcia@example.org) "
            "titulado «Visita cliente Bergen». Viaje nacional, 1 día (12 de marzo de 2026). "
            "Gastos: billete de tren 950 NOK, almuerzo 280 NOK. "
            "Envíe el informe después del registro."
        ),
        "expected_fields": {
            "employee_email": "elena.garcia@example.org",
            "title": "Visita cliente Bergen",
            "isForeignTravel": False,
            "costs": [
                {"description": "Billete de tren", "amount": 950},
                {"description": "Almuerzo", "amount": 280},
            ],
            "delivered": True,
        },
        "failure_mode_tested": "domestic_spanish_simple",
    },

    # ════════════════════════════════════════════════════════════════
    #  SALARY  (scoring 75%)
    # ════════════════════════════════════════════════════════════════

    # 7. Basic salary processing
    {
        "family": "salary",
        "language": "nb",
        "difficulty": "medium",
        "prompt": (
            "Kjør lønn for mars 2026 for Ola Nordmann (ola.nordmann@example.org). "
            "Grunnlønn 45000 kr. Skattetrekk 35%. "
            "Bruk konto 5000 for lønnskostnad, 2780 for skattetrekk, og 1920 for netto utbetaling."
        ),
        "expected_fields": {
            "employee_email": "ola.nordmann@example.org",
            "baseSalary": 45000,
            "taxRate": 0.35,
            "accounts": {"salary_cost": 5000, "tax": 2780, "bank": 1920},
        },
        "failure_mode_tested": "basic_salary_accounts",
    },

    # 8. Salary with overtime + bonus
    {
        "family": "salary",
        "language": "en",
        "difficulty": "hard",
        "prompt": (
            "Process payroll for March 2026 for Emma Wilson (emma.wilson@example.org). "
            "Base salary: 52000 NOK. Overtime supplement: 8500 NOK (account 5020). "
            "Bonus: 15000 NOK (account 5080). Tax deduction 33% on total. "
            "Net payment to bank account 1920."
        ),
        "expected_fields": {
            "employee_email": "emma.wilson@example.org",
            "baseSalary": 52000,
            "overtime": 8500,
            "bonus": 15000,
            "totalGross": 75500,
            "taxRate": 0.33,
            "accounts": {
                "salary_cost": 5000,
                "overtime_cost": 5020,
                "bonus_cost": 5080,
                "tax": 2780,
                "bank": 1920,
            },
        },
        "failure_mode_tested": "salary_with_supplements",
    },

    # 9. French salary
    {
        "family": "salary",
        "language": "fr",
        "difficulty": "medium",
        "prompt": (
            "Exécutez la paie de mars 2026 pour Jean Martin (jean.martin@example.org). "
            "Salaire de base : 48000 NOK. Déduction fiscale : 30%. "
            "Comptabilisez sur les comptes : 5000 (coût salarial), 2780 (impôt retenu), "
            "1920 (versement net en banque)."
        ),
        "expected_fields": {
            "employee_email": "jean.martin@example.org",
            "baseSalary": 48000,
            "taxRate": 0.30,
            "accounts": {"salary_cost": 5000, "tax": 2780, "bank": 1920},
        },
        "failure_mode_tested": "salary_french_basic",
    },

    # 10. Portuguese salary with bonus
    {
        "family": "salary",
        "language": "pt",
        "difficulty": "hard",
        "prompt": (
            "Processe o salário de março de 2026 para Ana Silva (ana.silva@example.org). "
            "Salário base: 55000 NOK. Bônus de desempenho: 12000 NOK (conta 5080). "
            "Dedução fiscal: 38% sobre o total. Pagamento líquido na conta 1920."
        ),
        "expected_fields": {
            "employee_email": "ana.silva@example.org",
            "baseSalary": 55000,
            "bonus": 12000,
            "totalGross": 67000,
            "taxRate": 0.38,
            "accounts": {"salary_cost": 5000, "bonus_cost": 5080, "tax": 2780, "bank": 1920},
        },
        "failure_mode_tested": "salary_portuguese_bonus",
    },

    # 11. Nynorsk salary
    {
        "family": "salary",
        "language": "nn",
        "difficulty": "medium",
        "prompt": (
            "Køyr løn for mars 2026 for Lars Bergstøl (lars.bergstol@example.org). "
            "Grunnløn 42000 kr. Skattetrekk 32%. "
            "Bruk konto 5000, 2780 og 1920."
        ),
        "expected_fields": {
            "employee_email": "lars.bergstol@example.org",
            "baseSalary": 42000,
            "taxRate": 0.32,
            "accounts": {"salary_cost": 5000, "tax": 2780, "bank": 1920},
        },
        "failure_mode_tested": "salary_nynorsk_basic",
    },

    # ════════════════════════════════════════════════════════════════
    #  TIMESHEET  (scoring 56-81%)
    # ════════════════════════════════════════════════════════════════

    # 12. Log hours + invoice (combined flow)
    {
        "family": "timesheet",
        "language": "nb",
        "difficulty": "hard",
        "prompt": (
            "Registrer 16 timer for Ingrid Svendsen (ingrid.svendsen@example.org) "
            "på aktiviteten «Utvikling» i prosjektet «Nettbutikk» for Fjelltopp AS "
            "(org.nr 934567821). Timesats 1200 kr. "
            "Fakturer deretter kunden for de registrerte timene."
        ),
        "expected_fields": {
            "employee_email": "ingrid.svendsen@example.org",
            "hours": 16,
            "activityName": "Utvikling",
            "projectName": "Nettbutikk",
            "customerName": "Fjelltopp AS",
            "customerOrgNumber": "934567821",
            "hourlyRate": 1200,
            "shouldInvoice": True,
        },
        "failure_mode_tested": "timesheet_and_invoice_combined",
    },

    # 13. Simple hour logging (no invoice)
    {
        "family": "timesheet",
        "language": "en",
        "difficulty": "easy",
        "prompt": (
            "Log 8 hours for Sarah Johnson (sarah.johnson@example.org) "
            "on the activity \"Consulting\" in the project \"ERP Migration\" "
            "for Nordic Solutions AS (org no. 912345678). "
            "Date: March 17, 2026. Hourly rate 950 NOK."
        ),
        "expected_fields": {
            "employee_email": "sarah.johnson@example.org",
            "hours": 8,
            "activityName": "Consulting",
            "projectName": "ERP Migration",
            "customerName": "Nordic Solutions AS",
            "customerOrgNumber": "912345678",
            "hourlyRate": 950,
            "date": "2026-03-17",
            "shouldInvoice": False,
        },
        "failure_mode_tested": "timesheet_hours_only",
    },

    # 14. German - hours + invoice
    {
        "family": "timesheet",
        "language": "de",
        "difficulty": "hard",
        "prompt": (
            "Registrieren Sie 12 Stunden für Michael Weber (michael.weber@example.org) "
            "auf der Aktivität «Design» im Projekt «App-Entwicklung» "
            "für Brightstone GmbH (Org.-Nr. 987654321). Stundensatz 1100 NOK. "
            "Erstellen Sie danach eine Projektrechnung an den Kunden."
        ),
        "expected_fields": {
            "employee_email": "michael.weber@example.org",
            "hours": 12,
            "activityName": "Design",
            "projectName": "App-Entwicklung",
            "customerName": "Brightstone GmbH",
            "customerOrgNumber": "987654321",
            "hourlyRate": 1100,
            "shouldInvoice": True,
        },
        "failure_mode_tested": "timesheet_german_invoice",
    },

    # 15. Issue #21 edge case: "timer på prosjektet" must go to timesheet, not project
    {
        "family": "timesheet",
        "language": "nb",
        "difficulty": "medium",
        "prompt": (
            "Registrer timer på prosjektet «Datamigrering» for Torstein Eide "
            "(torstein.eide@example.org). 10 timer på aktiviteten «Testing» "
            "for kunden Havneby AS (org.nr 876543210). Timesats 850 kr."
        ),
        "expected_fields": {
            "employee_email": "torstein.eide@example.org",
            "hours": 10,
            "activityName": "Testing",
            "projectName": "Datamigrering",
            "customerName": "Havneby AS",
            "customerOrgNumber": "876543210",
            "hourlyRate": 850,
            "shouldInvoice": False,
        },
        "failure_mode_tested": "planner_timesheet_vs_project_issue21",
    },

    # 16. Spanish timesheet
    {
        "family": "timesheet",
        "language": "es",
        "difficulty": "medium",
        "prompt": (
            "Registre 14 horas para Carlos López (carlos.lopez@example.org) "
            "en la actividad «Desarrollo» del proyecto «Portal Web» "
            "para Soluciones Nórdicas SL (nº org. 923456789). "
            "Tarifa por hora: 1050 NOK. Facture luego al cliente."
        ),
        "expected_fields": {
            "employee_email": "carlos.lopez@example.org",
            "hours": 14,
            "activityName": "Desarrollo",
            "projectName": "Portal Web",
            "customerName": "Soluciones Nórdicas SL",
            "customerOrgNumber": "923456789",
            "hourlyRate": 1050,
            "shouldInvoice": True,
        },
        "failure_mode_tested": "timesheet_spanish_invoice",
    },

    # ════════════════════════════════════════════════════════════════
    #  COST ANALYSIS  (scoring 28-38%)
    # ════════════════════════════════════════════════════════════════

    # 17. Full cost analysis (two-month comparison + project + activity)
    {
        "family": "cost_analysis",
        "language": "nb",
        "difficulty": "hard",
        "prompt": (
            "Totalkostnadene i selskapet har økt betydelig fra januar til februar 2026. "
            "Analyser hovedboken og finn de tre kontoene med størst kostnadsøkning. "
            "For hver av de tre kontoene: opprett et internt prosjekt med kontonavnet, "
            "og opprett en aktivitet med samme navn. "
            "Tilordne Markus Berg (markus.berg@example.org) som prosjektleder."
        ),
        "expected_fields": {
            "months_compared": ["2026-01", "2026-02"],
            "account_range": "4000-9999",
            "top_n": 3,
            "create_projects": True,
            "create_activities": True,
            "project_manager_email": "markus.berg@example.org",
            "isInternal": True,
        },
        "failure_mode_tested": "full_cost_analysis_nb",
    },

    # 18. Cost analysis in English
    {
        "family": "cost_analysis",
        "language": "en",
        "difficulty": "hard",
        "prompt": (
            "Total costs increased significantly from January to February 2026. "
            "Analyze the general ledger and identify the three expense accounts "
            "(4000-9999 range) with the largest cost increase between the two months. "
            "For each account, create an internal project named after the account, "
            "and create an activity with the same name as the project. "
            "Assign Lena Johansen (lena.johansen@example.org) as project manager."
        ),
        "expected_fields": {
            "months_compared": ["2026-01", "2026-02"],
            "account_range": "4000-9999",
            "top_n": 3,
            "create_projects": True,
            "create_activities": True,
            "project_manager_email": "lena.johansen@example.org",
            "isInternal": True,
        },
        "failure_mode_tested": "full_cost_analysis_en",
    },

    # 19. German cost analysis
    {
        "family": "cost_analysis",
        "language": "de",
        "difficulty": "hard",
        "prompt": (
            "Die Gesamtkosten sind von Januar bis Februar 2026 erheblich gestiegen. "
            "Analysieren Sie das Hauptbuch und finden Sie die drei Aufwandskonten "
            "(4000-9999) mit dem größten Kostenanstieg. "
            "Erstellen Sie für jedes Konto ein internes Projekt mit dem Kontonamen "
            "und eine Aktivität mit demselben Namen. "
            "Weisen Sie Erik Haugen (erik.haugen@example.org) als Projektleiter zu."
        ),
        "expected_fields": {
            "months_compared": ["2026-01", "2026-02"],
            "account_range": "4000-9999",
            "top_n": 3,
            "create_projects": True,
            "create_activities": True,
            "project_manager_email": "erik.haugen@example.org",
            "isInternal": True,
        },
        "failure_mode_tested": "full_cost_analysis_de",
    },

    # 20. Nynorsk cost analysis (Issue F19: previously misclassified)
    {
        "family": "cost_analysis",
        "language": "nn",
        "difficulty": "hard",
        "prompt": (
            "Kostnadene i selskapet har auka mykje frå januar til februar 2026. "
            "Analyser hovudboka og finn dei tre kontoane med størst kostnadsauke "
            "i intervallet 4000-9999. "
            "For kvar konto: opprett eit internt prosjekt med kontonamnet, "
            "og opprett ein aktivitet med same namn. "
            "Sett Silje Bakken (silje.bakken@example.org) som prosjektleiar."
        ),
        "expected_fields": {
            "months_compared": ["2026-01", "2026-02"],
            "account_range": "4000-9999",
            "top_n": 3,
            "create_projects": True,
            "create_activities": True,
            "project_manager_email": "silje.bakken@example.org",
            "isInternal": True,
        },
        "failure_mode_tested": "cost_analysis_nynorsk_classification",
    },

    # 21. Portuguese cost analysis
    {
        "family": "cost_analysis",
        "language": "pt",
        "difficulty": "hard",
        "prompt": (
            "Os custos totais da empresa aumentaram significativamente de janeiro a fevereiro de 2026. "
            "Analise o razão geral e identifique as três contas de despesas (4000-9999) "
            "com o maior aumento de custos. "
            "Para cada conta, crie um projeto interno com o nome da conta "
            "e crie uma atividade com o mesmo nome. "
            "Atribua João Oliveira (joao.oliveira@example.org) como gerente de projeto."
        ),
        "expected_fields": {
            "months_compared": ["2026-01", "2026-02"],
            "account_range": "4000-9999",
            "top_n": 3,
            "create_projects": True,
            "create_activities": True,
            "project_manager_email": "joao.oliveira@example.org",
            "isInternal": True,
        },
        "failure_mode_tested": "cost_analysis_portuguese",
    },

    # ════════════════════════════════════════════════════════════════
    #  EXTRA: Edge cases from ISSUES.md
    # ════════════════════════════════════════════════════════════════

    # 22. Issue #25: "formation" in description should NOT trigger food VAT
    {
        "family": "invoice",
        "language": "fr",
        "difficulty": "hard",
        "prompt": (
            "Créez une facture pour le client Formation Expert SARL (nº org. 945678123) "
            "avec une ligne : «Formation en gestion de projet» "
            "quantité 1, prix unitaire 18500 NOK HT, TVA standard 25%. "
            "Envoyez la facture par e-mail."
        ),
        "expected_fields": {
            "customerName": "Formation Expert SARL",
            "customerOrgNumber": "945678123",
            "lines": [
                {
                    "description": "Formation en gestion de projet",
                    "unitPrice": 18500,
                    "vatRate": 25,
                    "count": 1,
                },
            ],
            "shouldSend": True,
        },
        "failure_mode_tested": "formation_false_food_vat_issue25",
    },

    # 23. Issue #20: "Produktlinje" dimension → voucher, not product
    {
        "family": "voucher",
        "language": "nb",
        "difficulty": "medium",
        "prompt": (
            "Opprett en egendefinert regnskapsdimensjon «Produktlinje» med verdiene "
            "«Standard» og «Premium». Deretter bokfør et bilag datert 15. mars 2026: "
            "debet konto 4000 (12000 kr) med dimensjon Produktlinje=Standard, "
            "kredit konto 1920 (12000 kr)."
        ),
        "expected_fields": {
            "dimensionName": "Produktlinje",
            "dimensionValues": ["Standard", "Premium"],
            "voucher_date": "2026-03-15",
            "postings": [
                {"account": 4000, "debit": 12000, "dimension": "Standard"},
                {"account": 1920, "credit": 12000},
            ],
        },
        "failure_mode_tested": "produktlinje_dimension_planner_issue20",
    },

    # 24. Issue #21: "timer på prosjektet" → timesheet not project
    {
        "family": "timesheet",
        "language": "nb",
        "difficulty": "medium",
        "prompt": (
            "Registrer timer på prosjektet «CRM-Integrasjon» for Helene Strand "
            "(helene.strand@example.org). 20 timer på aktiviteten «Konsultering» "
            "for kunden Vestland Shipping AS (org.nr 945123678). "
            "Timesats 1150 kr. Fakturer kunden etterpå."
        ),
        "expected_fields": {
            "employee_email": "helene.strand@example.org",
            "hours": 20,
            "activityName": "Konsultering",
            "projectName": "CRM-Integrasjon",
            "customerName": "Vestland Shipping AS",
            "customerOrgNumber": "945123678",
            "hourlyRate": 1150,
            "shouldInvoice": True,
        },
        "failure_mode_tested": "planner_timer_prosjektet_issue21",
    },

    # ════════════════════════════════════════════════════════════════
    #  LIVE FAILURE REPRODUCTIONS (from /tmp/accounting-logs)
    # ════════════════════════════════════════════════════════════════

    # 25. Issue #27: French supplier invoice misclassified as "invoice"
    #     Live: 20260322_083514 — planner routes "facture fournisseur" to invoice family.
    #     Result: POST /incomingInvoice → 403, falls back to voucher. Should be supplier family.
    {
        "family": "supplier",
        "language": "fr",
        "difficulty": "hard",
        "prompt": (
            "Vous avez recu une facture fournisseur (voir PDF ci-joint). "
            "Enregistrez la facture dans Tripletex. "
            "Creez le fournisseur s'il n'existe pas. "
            "Utilisez le bon compte de charges et la TVA deductible."
        ),
        "expected_fields": {
            "supplierName": "Lumière SARL",
            "supplierOrgNumber": "908587022",
            "voucherFallback": True,
            "chargeAccount": "6540",
        },
        "failure_mode_tested": "french_supplier_invoice_misclassified_issue27",
    },

    # 26. Issue #28: Complex project lifecycle — hourlyRateModel enum + incomingInvoice 403 + duplicates
    #     Live: 20260322_084427 — 4 errors in 22 API calls.
    #     Errors: hourlyRateModel "FIXED_HOURLY_RATE" → 422, incomingInvoice → 403,
    #     duplicate hourlyRates retry → 409, duplicate timesheet → 409.
    {
        "family": "project",
        "language": "en",
        "difficulty": "hard",
        "prompt": (
            "Execute the complete project lifecycle for 'System Upgrade Greenfield' "
            "(Greenfield Ltd, org no. 873288949): "
            "1) The project has a budget of 206300 NOK. "
            "2) Log time: Oliver Wilson (project manager, oliver.wilson@example.org) "
            "36 hours and Victoria Taylor (consultant, victoria.taylor@example.org) 150 hours. "
            "3) Register supplier cost of 41300 NOK from Ironbridge Ltd (org no. 913777255). "
            "4) Create a customer invoice for the project."
        ),
        "expected_fields": {
            "projectName": "System Upgrade Greenfield",
            "customerName": "Greenfield Ltd",
            "customerOrgNumber": "873288949",
            "budget": 206300,
            "employees": [
                {"email": "oliver.wilson@example.org", "hours": 36, "role": "project_manager"},
                {"email": "victoria.taylor@example.org", "hours": 150, "role": "consultant"},
            ],
            "supplierCost": 41300,
            "supplierName": "Ironbridge Ltd",
            "supplierOrgNumber": "913777255",
            "shouldInvoice": True,
        },
        "failure_mode_tested": "project_lifecycle_hourlyrate_enum_incoming403_duplicates_issue28",
    },

    # 27. Issue #31: French overdue invoice + reminder fee — createReminder 422 x3
    #     Live: 20260322_091047 — LLM doesn't know createReminder needs type+date params.
    #     Also GET /invoice without date params → 422 x2. System needs action-layer guard.
    {
        "family": "invoice",
        "language": "fr",
        "difficulty": "hard",
        "prompt": (
            "L'un de vos clients a une facture en retard. "
            "Trouvez la facture en retard et enregistrez des frais de rappel de 70 NOK. "
            "Debit creances clients (1500), credit revenus de rappel (3400). "
            "Créez également une facture pour les frais de rappel au client et envoyez-la. "
            "De plus, enregistrez un paiement partiel de 5000 NOK sur la facture en retard."
        ),
        "expected_fields": {
            "reminderFee": 70,
            "reminderAccounts": {"debit": 1500, "credit": 3400},
            "partialPayment": 5000,
            "shouldSend": True,
            "createReminder": True,
        },
        "failure_mode_tested": "overdue_reminder_fee_createreminder_422_issue31",
    },

    # 28. Portuguese multi-VAT invoice — all lines got vatType id=3 (25%) instead of correct types
    #     Live: 20260322_091349 — 0 errors but VAT types wrong: should be 25%, 15% food, 0% exempt.
    {
        "family": "invoice",
        "language": "pt",
        "difficulty": "hard",
        "prompt": (
            "Crie uma fatura para o cliente Floresta Lda (org. nº 944182802) "
            "com três linhas de produto: "
            "Relatório de análise (2039) a 19600 NOK com 25% IVA, "
            "Design web (3304) a 12450 NOK com 15% IVA (alimentos), "
            "e Licença de software (1599) a 9150 NOK com 0% IVA (isento)."
        ),
        "expected_fields": {
            "customerName": "Floresta Lda",
            "customerOrgNumber": "944182802",
            "lines": [
                {"description": "Relatório de análise", "unitPrice": 19600, "vatRate": 25, "productNumber": "2039"},
                {"description": "Design web", "unitPrice": 12450, "vatRate": 15, "productNumber": "3304"},
                {"description": "Licença de software", "unitPrice": 9150, "vatRate": 0, "productNumber": "1599"},
            ],
        },
        "failure_mode_tested": "multi_vat_rate_portuguese_all_lines_got_25pct",
    },

    # 29. Portuguese employee from PDF — planner misclassifies as "department"
    #     Live: 20260322_091356 — "departamento" in prompt triggers department family.
    #     Execution succeeded anyway, but planner should route to employee.
    {
        "family": "employee",
        "language": "pt",
        "difficulty": "hard",
        "prompt": (
            "Voce recebeu um contrato de trabalho (ver PDF anexo). "
            "Crie o funcionario no Tripletex com todos os detalhes do contrato: "
            "numero de identidade nacional, data de nascimento, departamento, "
            "codigo de ocupacao, salario, percentagem de emprego e data de inicio."
        ),
        "expected_fields": {
            "firstName": "Mariana",
            "lastName": "Costa",
            "email": "mariana.costa@example.org",
            "dateOfBirth": "1999-10-22",
            "nationalIdentityNumber": "22109922602",
            "department": "Kundeservice",
            "occupationCode": "2511",
            "annualSalary": 510000,
            "employmentPercentage": 100,
            "startDate": "2026-06-08",
        },
        "failure_mode_tested": "portuguese_employee_misclassified_as_department",
    },

    # 30. Cost analysis activity duplicate name "General" — 422 x3
    #     Live: 20260322_094249 — agent tries to create activity named "General" which already exists.
    #     LLM retries 3x with same name. Should check existing or use account-specific names.
    {
        "family": "cost_analysis",
        "language": "en",
        "difficulty": "hard",
        "prompt": (
            "Total costs increased significantly from January to February 2026. "
            "Analyze the general ledger and identify the three expense accounts "
            "with the largest increase in amount. "
            "Create an internal project for each of the three accounts using the account name. "
            "Also create an activity for each project."
        ),
        "expected_fields": {
            "months_compared": ["2026-01", "2026-02"],
            "account_range": "4000-9999",
            "top_n": 3,
            "create_projects": True,
            "create_activities": True,
            "isInternal": True,
        },
        "failure_mode_tested": "cost_analysis_activity_duplicate_name_general",
    },

    # 31. Bank reconciliation — GET /supplierInvoice missing date params → 422 x6
    #     Live: 20260322_094334 — LLM searches supplier invoices without required date params.
    #     Client-level intercept missing for /supplierInvoice (only /invoice was covered).
    {
        "family": "bank_reconciliation",
        "language": "es",
        "difficulty": "hard",
        "prompt": (
            "Concilia el extracto bancario (CSV adjunto) con las facturas abiertas en Tripletex. "
            "Relaciona los pagos entrantes con las facturas de clientes "
            "y los pagos salientes con las facturas de proveedores. "
            "Maneja los pagos parciales correctamente."
        ),
        "expected_fields": {
            "matchIncoming": True,
            "matchOutgoing": True,
            "handlePartialPayments": True,
        },
        "failure_mode_tested": "bank_recon_supplier_invoice_missing_date_params",
    },

    # 32. Employee from PDF — Spanish, 0 errors but only 3/8 score
    #     Live: 20260322_094937 — all fields extracted correctly but scorer expects more.
    #     Agent created: employee, department, employment, employment/details, standardTime.
    #     Scored 3/8 despite doing everything right. Scorer gap investigation needed.
    {
        "family": "employee",
        "language": "es",
        "difficulty": "hard",
        "prompt": (
            "Has recibido un contrato de trabajo (ver PDF adjunto). "
            "Crea el empleado en Tripletex con todos los datos del contrato: "
            "numero de identidad, fecha de nacimiento, departamento, "
            "codigo de ocupacion, salario, porcentaje de empleo y fecha de inicio."
        ),
        "expected_fields": {
            "firstName": "Lucía",
            "lastName": "Martínez",
            "email": "lucia.martinez@example.org",
            "dateOfBirth": "1998-04-17",
            "nationalIdentityNumber": "17049848676",
            "department": "Markedsføring",
            "occupationCode": "1211",
            "annualSalary": 550000,
            "employmentPercentage": 100,
            "startDate": "2026-06-15",
            "hoursPerDay": 7.5,
            "userType": "EXTENDED",
        },
        "failure_mode_tested": "employee_pdf_spanish_all_fields_but_low_score",
    },

    # 33. Simple employee — English, 0 errors but only 2/7 score
    #     Live: 20260322_095248 — simple prompt, agent guessed department "Drift" not in prompt.
    #     Only 5 API calls. Missing employment details because prompt didn't mention them.
    {
        "family": "employee",
        "language": "en",
        "difficulty": "easy",
        "prompt": (
            "We have a new employee named Thomas Harris, born 4. June 1991. "
            "Please create them as an employee with email thomas.harris@example.org "
            "and start date 6. October 2026."
        ),
        "expected_fields": {
            "firstName": "Thomas",
            "lastName": "Harris",
            "email": "thomas.harris@example.org",
            "dateOfBirth": "1991-06-04",
            "startDate": "2026-10-06",
        },
        "failure_mode_tested": "simple_employee_low_score_missing_fields",
    },

    # 34. French supplier invoice with PDF — incomingInvoice 403, voucher fallback scores 3/8
    #     Live: 20260322_095237 — planner correctly routes to supplier, but voucher fallback
    #     only gets partial score. Need to understand what scorer fields the voucher misses.
    {
        "family": "supplier",
        "language": "fr",
        "difficulty": "hard",
        "prompt": (
            "Vous avez recu une facture fournisseur (voir PDF ci-joint). "
            "Enregistrez la facture dans Tripletex. "
            "Creez le fournisseur s'il n'existe pas. "
            "Utilisez le bon compte de charges et la TVA deductible."
        ),
        "expected_fields": {
            "supplierName": "Forêt SARL",
            "supplierOrgNumber": "900969147",
            "invoiceNumber": "INV-2026-1881",
            "chargeAccount": "6540",
            "amountIncludingVat": 30562,
            "voucherFallback": True,
        },
        "failure_mode_tested": "supplier_invoice_403_voucher_fallback_partial_score",
    },

    # 35. German supplier invoice — planner misroutes "Lieferantenrechnung" to invoice
    #     Live: 20260322_100606 — "Rechnung" keyword wins for invoice over "Lieferant" for supplier.
    #     FIXED: added "Lieferantenrechnung" as compound supplier keyword.
    {
        "family": "supplier",
        "language": "de",
        "difficulty": "hard",
        "prompt": (
            "Sie haben eine Lieferantenrechnung erhalten (siehe beigefügte PDF). "
            "Registrieren Sie die Rechnung in Tripletex. "
            "Erstellen Sie den Lieferanten, falls er nicht existiert. "
            "Verwenden Sie das richtige Aufwandskonto und die abzugsfähige Mehrwertsteuer."
        ),
        "expected_fields": {
            "supplierCreated": True,
            "chargeAccount": True,
            "vatDeductible": True,
            "voucherFallback": True,
        },
        "failure_mode_tested": "german_supplier_invoice_lieferantenrechnung_misroute",
    },

    # 36. Issue #26: Payment reversal creates TWO invoices instead of one
    #     LLM ignores action-layer auto-create signal, creates second invoice.
    {
        "family": "invoice",
        "language": "nb",
        "difficulty": "hard",
        "prompt": (
            "Kunden Nordlys AS (org.nr 912345678) betalte faktura #1 på 25000 kr, "
            "men betalingen ble returnert av banken. "
            "Reverser betalingen på den eksisterende fakturaen."
        ),
        "expected_fields": {
            "customerName": "Nordlys AS",
            "customerOrgNumber": "912345678",
            "invoiceCount": 1,
            "paymentReversed": True,
        },
        "failure_mode_tested": "payment_reversal_double_invoice_issue26",
    },

    # 37. Issue #24: Fixed-price project — LLM uses hourlyRates instead of project.fixedprice
    {
        "family": "project",
        "language": "nb",
        "difficulty": "hard",
        "prompt": (
            "Opprett et fastprisprosjekt «Nettside Redesign» for kunden Fjordtech AS "
            "(org.nr 987654321) med fastpris 150000 kr. "
            "Prosjektleder er Erik Hansen (erik.hansen@example.org). "
            "Fakturer 50% av fastprisen som delbetaling."
        ),
        "expected_fields": {
            "projectName": "Nettside Redesign",
            "customerName": "Fjordtech AS",
            "customerOrgNumber": "987654321",
            "fixedPrice": 150000,
            "invoicePercent": 50,
            "projectManagerEmail": "erik.hansen@example.org",
            "isFixedPrice": True,
        },
        "failure_mode_tested": "fixed_price_project_hourlyrates_bypass_issue24",
    },

    # 38. Issue #23: Order→invoice conversion missing invoiceDate
    {
        "family": "invoice",
        "language": "en",
        "difficulty": "hard",
        "prompt": (
            "Create an order for customer Greenfield Ltd (org no. 873288949) "
            "with one line: 'Consulting services' quantity 10, unit price 1500 NOK, 25% VAT. "
            "Then convert the order to an invoice and send it to the customer."
        ),
        "expected_fields": {
            "customerName": "Greenfield Ltd",
            "customerOrgNumber": "873288949",
            "orderCreated": True,
            "orderConvertedToInvoice": True,
            "invoiceDate": True,
            "shouldSend": True,
        },
        "failure_mode_tested": "order_to_invoice_missing_invoicedate_issue23",
    },
]


def get_prompts(family: str = None, difficulty: str = None, language: str = None) -> list[dict]:
    """Filter adversarial prompts by family, difficulty, or language."""
    result = ADVERSARIAL_PROMPTS
    if family:
        result = [p for p in result if p["family"] == family]
    if difficulty:
        result = [p for p in result if p["difficulty"] == difficulty]
    if language:
        result = [p for p in result if p["language"] == language]
    return result


if __name__ == "__main__":
    import json
    from collections import Counter

    families = Counter(p["family"] for p in ADVERSARIAL_PROMPTS)
    languages = Counter(p["language"] for p in ADVERSARIAL_PROMPTS)
    difficulties = Counter(p["difficulty"] for p in ADVERSARIAL_PROMPTS)

    print(f"Total prompts: {len(ADVERSARIAL_PROMPTS)}")
    print(f"\nBy family: {dict(families.most_common())}")
    print(f"By language: {dict(languages.most_common())}")
    print(f"By difficulty: {dict(difficulties.most_common())}")
    print(f"\nFailure modes tested:")
    for p in ADVERSARIAL_PROMPTS:
        print(f"  [{p['family']}] {p['failure_mode_tested']}")
