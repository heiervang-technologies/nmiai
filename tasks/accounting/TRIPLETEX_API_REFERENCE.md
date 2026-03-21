# Tripletex API v2 - Sandbox Reference

Probed from sandbox `kkpqfuj-amager.tripletex.dev` on 2026-03-21.

## Auth
```
Basic auth: username=0, password=<session_token>
Base URL: https://<sandbox>.tripletex.dev/v2
```

## Pre-populated Sandbox Data

### Employees (5 found)
| ID | Name | Email |
|----|------|-------|
| 18450063 | Marksverdhei b3d2ff90 | marksverdhei@proton.me |
| 18575565 | Kari Hansen | kari.hansen@firma.no |
| 18626885 | Kari Nordmann | kari@example.com |
| 18626927 | Ola Nordmann | ola@test.no |
| 18627425 | Erik Hansen | erik@test.no |

### Departments
| ID | Number | Name |
|----|--------|------|
| 842220 | (none) | Avdeling |
| 932601 | 200 | Salg |
| 932978 | 100 | Salg |
| 932979 | 300 | HR |
| 933148 | 500 | IT |

### Activities
| ID | Name | Type |
|----|------|------|
| 5604693 | Administrasjon | GENERAL_ACTIVITY |
| 5604694 | Ferie | GENERAL_ACTIVITY |
| 5604695 | Prosjektadministrasjon | PROJECT_GENERAL_ACTIVITY |
| 5604696 | Fakturerbart arbeid | PROJECT_GENERAL_ACTIVITY |
| 5892559 | utvikling | GENERAL_ACTIVITY |

### Payment Types
| ID | Description | Context |
|----|-------------|---------|
| 33009453 | Kontant | Invoice payment |
| 33009454 | Betalt til bank | Invoice payment |
| 33009412 | Privat utlegg | Travel expense |

### VAT Types (key ones)
| ID | Percentage | Name | Use For |
|----|-----------|------|---------|
| 1 | 25% | Fradrag inngaende avgift, hoy sats | Expense vouchers |
| 3 | 25% | Utgaende avgift, hoy sats | Sales/invoices |
| 5 | 0% | Ingen utgaende avgift (innenfor) | Zero-rated sales |
| 6 | 0% | Ingen utgaende avgift (utenfor) | Products (only valid for products in this sandbox!) |
| 11 | 15% | Fradrag inngaende middels | Food expenses |
| 12 | 12% | Fradrag inngaende lav | Transport expenses |
| 31 | 15% | Utgaende middels | Food sales |
| 32 | 12% | Utgaende lav | Transport sales |

### Key Account Numbers (Norwegian Standard)
| Number | Name | Use |
|--------|------|-----|
| 1500 | Kundefordringer | Accounts receivable |
| 1920 | Bankinnskudd | Bank deposits |
| 2780 | Paloept arbeidsgiveravgift | Employer tax accrual |
| 3000 | Salgsinntekt, avgiftspliktig | Sales revenue (taxable) |
| 5000 | Loenn til ansatte | Salary expense |
| 6300 | Leie lokale | Office rent |
| 6590 | Annet driftsmateriale | Other operating materials |
| 7140 | Reisekostnad | Travel expense |

### Travel Expense Rate Categories (PER_DIEM)
| ID | Name |
|----|------|
| 2 | Dagsreise 5-9 timer - innland |
| 3 | Dagsreise 9-12 timer - innland |
| 4 | Dagsreise over 12 timer - innland |
| 10 | Overnatting 8-12 timer - innland |
| 11 | Overnatting over 12 timer - innland |

**CRITICAL: `GET /travelExpense/rate` returns EMPTY in this sandbox.**
Use `rateCategory` (above) instead of `rateType` for per diem.

## Endpoint Quick Reference

### POST /employee
```json
{
  "firstName": "Ola",
  "lastName": "Nordmann",
  "email": "ola@example.com",
  "userType": "STANDARD",
  "department": {"id": 842220},
  "dateOfBirth": "1990-01-15"
}
```
- `userType`: STANDARD, EXTENDED (admin), NO_ACCESS
- `department.id` required — GET /department first
- Admin access = `userType: "EXTENDED"`

### POST /customer
```json
{
  "name": "Firma AS",
  "organizationNumber": "123456789",
  "email": "post@firma.no",
  "postalAddress": {"addressLine1": "Gate 1", "postalCode": "0150", "city": "Oslo"}
}
```

### POST /product
```json
{"name": "Produkt", "priceExcludingVatCurrency": 1500.0}
```
- Do NOT send `vatType` unless confirmed working — defaults to id=6
- Field is `priceExcludingVatCurrency` not `priceExcludingVat`

### POST /invoice
```json
{
  "invoiceDate": "2026-03-21",
  "invoiceDueDate": "2026-04-21",
  "orders": [{
    "customer": {"id": 123},
    "orderDate": "2026-03-21",
    "deliveryDate": "2026-03-21",
    "orderLines": [{
      "description": "Service",
      "count": 1,
      "unitPriceExcludingVatCurrency": 1500.0,
      "vatType": {"id": 3}
    }]
  }]
}
```
- Company must have `bankAccountNumber` set first
- Each orderLine MUST have `vatType`

### POST /ledger/voucher
```json
{
  "date": "2026-03-21",
  "description": "Journal entry",
  "postings": [
    {"row": 1, "account": {"id": 6300}, "amountGross": 5000.0},
    {"row": 2, "account": {"id": 1920}, "amountGross": -5000.0}
  ]
}
```
- Row numbers start at 1 (NOT 0)
- Positive = debit, negative = credit
- Must balance (sum to zero)

### POST /travelExpense
```json
{
  "employee": {"id": 18450063},
  "title": "Trip to Oslo",
  "travelDetails": {
    "departureDate": "2026-03-20",
    "returnDate": "2026-03-22",
    "departure": "Bergen",
    "destination": "Oslo"
  }
}
```
- MUST include `travelDetails` block
- Employee needs salary/travel entitlement (cannot be granted via API)

### POST /activity
```json
{"name": "Development", "activityType": "GENERAL_ACTIVITY"}
```
- `activityType` is REQUIRED (cannot be null)
- Values: GENERAL_ACTIVITY, PROJECT_SPECIFIC_ACTIVITY

### POST /timesheet/entry
```json
{
  "employee": {"id": 18450063},
  "project": {"id": 123},
  "activity": {"id": 5604696},
  "date": "2026-03-21",
  "hours": 8
}
```
- Activity must be linked to project first
- Employee needs timesheet entitlement

### GET endpoints requiring date params
| Endpoint | Required Params |
|----------|----------------|
| GET /invoice | invoiceDateFrom, invoiceDateTo |
| GET /ledger/posting | dateFrom, dateTo |
| GET /ledger/voucher | dateFrom, dateTo |
| GET /order | orderDateFrom, orderDateTo |
