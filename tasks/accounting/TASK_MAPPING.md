# Task Mapping & Architecture Deep Dive

## Core Architecture Decision: Two Approaches

### Approach A: Structured Parser + Dispatch (Current)
```
Prompt → LLM extracts JSON {task_type, fields} → Dispatch to handler → API calls
```
**Pros:** Predictable, minimal API calls, easy to optimize efficiency bonus
**Cons:** Brittle - must anticipate every task variant, field mapping errors

### Approach B: Agentic LLM with Tools (Recommended)
```
Prompt → LLM with Tripletex API as tools → LLM decides which APIs to call → Done
```
**Pros:** Handles unknown tasks, adapts to any prompt variant, self-correcting
**Cons:** More API calls (hurts efficiency), slower, costs more LLM tokens

### Approach C: Hybrid (Best of Both)
```
Prompt → LLM classifies task type → If known type: structured handler (fast, efficient)
                                   → If unknown: fall back to agentic mode
```
**This is the recommended approach.** Known Tier 1 tasks get optimized handlers.
Unknown or complex tasks get agentic fallback.

---

## Corrected Tier Breakdown (from docs research)

### TIER 1 (x1): Foundational single-entity tasks
- Create employee, create customer, create invoice

### TIER 2 (x2): Multi-step workflows
- Invoice with payment, credit notes, project billing

### TIER 3 (x3): Complex scenarios (opens early Saturday)
- Bank reconciliation from CSV, error correction in ledger, year-end closing

### Rate Limits
| | Verified Teams | Unverified Teams |
|---|---|---|
| Concurrent submissions | 3 | 1 |
| Per task per day | 10 | 3 |

### Task Assignment
Each submission picks a task **weighted toward less-attempted types** - so you'll see new tasks more often.

### MCP Server
`claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp`

---

## Task Type Analysis

### TIER 1 Tasks (x1 multiplier) - Available NOW

#### 1. Create Employee
**Prompt pattern:** "Opprett en ansatt med navn [X], e-post [Y]..."
**API flow:**
```
POST /employee
{
  "firstName": "Ola",
  "lastName": "Nordmann",
  "email": "ola@example.com",
  "phoneNumberMobile": "12345678",
  "dateOfBirth": "1990-01-15"
}
```
**Checked fields:** employee found, firstName, lastName, email, role/admin flag
**Min API calls:** 1 (just POST)
**Notes:** May need to set `isAdministrator` or employment details

#### 2. Update Employee
**Prompt pattern:** "Oppdater [ansatt] sin e-post til [Y]..."
**API flow:**
```
GET /employee?firstName=Ola  (find existing)
PUT /employee/{id}           (update fields)
```
**Checked fields:** updated field values match
**Min API calls:** 2 (GET + PUT)
**Notes:** Fresh sandbox means employee may need creating first. Or maybe the sandbox comes pre-populated? Need to test.

#### 3. Create Customer
**Prompt pattern:** "Registrer en ny kunde: [Firma AS], org.nr [X]..."
**API flow:**
```
POST /customer
{
  "name": "Firma AS",
  "organizationNumber": "123456789",
  "email": "post@firma.no",
  "phoneNumber": "99887766",
  "postalAddress": {
    "addressLine1": "Gateveien 1",
    "postalCode": "0150",
    "city": "Oslo"
  }
}
```
**Checked fields:** customer found, name, org number, email, address fields
**Min API calls:** 1

#### 4. Create Product
**Prompt pattern:** "Opprett et produkt: [Navn], pris [X] kr eks. mva..."
**API flow:**
```
POST /product
{
  "name": "Konsulenttime",
  "number": "1001",
  "priceExcludingVat": 1200.00,
  "vatType": {"id": 3}  // Standard 25% MVA
}
```
**Checked fields:** product found, name, price, VAT type
**Min API calls:** 1
**Notes:** Need to know VAT type IDs. Common: id=3 (25% MVA), id=5 (15%), id=6 (0%)

#### 5. Create Department
**Prompt pattern:** "Opprett en avdeling: [Navn]..."
**API flow:**
```
POST /department
{
  "name": "Salg",
  "departmentNumber": "100"
}
```
**Min API calls:** 1

#### 6. Enable Module
**Prompt pattern:** "Aktiver modulen for [reiseregning/prosjekt]..."
**API flow:** Unknown - need to discover. Possibly:
```
PUT /company/modules
```
**Notes:** CRITICAL to figure out. Some tasks may require modules enabled first.

---

### TIER 2 Tasks (x2 multiplier) - Unlocks early Friday

#### 7. Create Invoice (via Order)
**Prompt pattern:** "Opprett en faktura til [kunde] for [produkt] x [antall]..."
**API flow (multi-step):**
```
1. POST /customer          (if not exists)
2. POST /product           (if not exists)
3. POST /order             (create order with lines)
   {
     "customer": {"id": customer_id},
     "orderDate": "2026-03-20",
     "deliveryDate": "2026-03-20",
     "orderLines": [
       {
         "product": {"id": product_id},
         "count": 5,
         "unitPriceExcludingVatCurrency": 1200.00
       }
     ]
   }
4. PUT /order/{id}/:invoice  (convert order to invoice)
   {
     "invoiceDate": "2026-03-20",
     "invoiceDueDate": "2026-04-20"
   }
```
**Checked fields:** invoice exists, correct customer, correct line items, amounts
**Min API calls:** 2-4 (depending on prerequisites)
**KEY GOTCHA:** Tripletex invoicing goes through orders! You don't POST to /invoice directly.

#### 8. Register Payment on Invoice
**Prompt pattern:** "Registrer betaling på faktura [X], beløp [Y]..."
**API flow:**
```
1. GET /invoice             (find the invoice)
2. POST /invoice/{id}/:payment
   {
     "paymentDate": "2026-03-20",
     "paymentType": {"id": payment_type_id},
     "amount": 6000.00,
     "amountCurrency": 6000.00
   }
```
**Notes:** Need to know paymentType IDs. GET /ledger/paymentTypeOut to discover.

#### 9. Create Travel Expense
**Prompt pattern:** "Registrer en reiseregning for [ansatt]..."
**API flow:**
```
POST /travelExpense
{
  "employee": {"id": emp_id},
  "title": "Reise til Bergen",
  "departureDate": "2026-03-15",
  "returnDate": "2026-03-16",
  "travelAdvance": 500.00
}
```
**Notes:** May need travel expense module enabled first!

#### 10. Create Project
**Prompt pattern:** "Opprett et prosjekt: [Navn], knyttet til kunde [X]..."
**API flow:**
```
POST /project
{
  "name": "Webside-redesign",
  "number": "P001",
  "projectManager": {"id": employee_id},
  "customer": {"id": customer_id},
  "startDate": "2026-03-01",
  "endDate": "2026-06-30",
  "isClosed": false
}
```
**Notes:** projectManager must be a valid employee. May need to find/create one.

#### 11. Delete Travel Expense
**Prompt pattern:** "Slett reiseregningen for [ansatt]..."
**API flow:**
```
GET /travelExpense          (find it)
DELETE /travelExpense/{id}
```

---

### TIER 3 Tasks (x3 multiplier) - Unlocks early Saturday

#### 12. Bank Reconciliation from CSV
**Prompt pattern:** "Avstem bankkontoen med vedlagt CSV-fil..."
**API flow (speculative):**
```
1. Decode base64 CSV file attachment
2. Parse CSV rows (date, description, amount)
3. GET /ledger/account (find bank account)
4. POST /bank/reconciliation or POST /ledger/voucher for each row
5. Match transactions to existing postings
```
**Notes:** This is complex - needs file parsing + accounting logic.
The agent must understand CSV structure and create correct journal entries.

#### 13. Error Correction in Ledger
**Prompt pattern:** "Korriger feil i hovedboken: bilag [X] har feil konto..."
**API flow (speculative):**
```
1. GET /ledger/voucher (find the incorrect voucher)
2. Create reversing entry (credit what was debited, vice versa)
3. Create correct entry
```
**Notes:** May need to understand double-entry bookkeeping.

#### 14. Year-End Closing
**Prompt pattern:** "Utfør årsavslutning for [year]..."
**API flow (speculative):**
```
1. Close all open periods
2. Transfer P&L to balance sheet
3. Create closing vouchers
```
**Notes:** Most complex task. Requires understanding of Norwegian accounting year-end procedures.

#### 15. Create Credit Note
**Prompt pattern:** "Opprett en kreditnota for faktura [X]..."
**API flow:**
```
GET /invoice                 (find the invoice)
POST /invoice/{id}/:createCreditNote
```

#### 16. Delete/Reverse Invoice
**Prompt pattern:** "Reverser/slett faktura [X]..."

---

## Scoring Example (from docs)

**"Create employee" task (max 10 points):**
| Check | Points |
|-------|--------|
| Employee found | 2 |
| Correct first name | 1 |
| Correct last name | 1 |
| Correct email | 1 |
| Administrator role assigned | 5 |

Note: Administrator role is worth **50% of the points** - easy to miss!

**Efficiency bonus example (Tier 2):**
| Scenario | Score |
|----------|-------|
| Failed all checks | 0.0 |
| 80% checks passed | 1.6 |
| Perfect, many errors | ~2.1 |
| Perfect, efficient, few errors | ~2.6 |
| Perfect, best efficiency, zero errors | 4.0 |

---

## Critical Unknowns to Resolve

1. **What's pre-populated in a fresh sandbox?**
   - Are there any employees, customers, products already?
   - What's the chart of accounts?
   - Are any modules enabled by default?

2. **VAT Type IDs** - Need to GET /ledger/vatType on a sandbox
   - Standard 25% MVA
   - Reduced 15%
   - Zero-rated
   - Exempt

3. **Payment Type IDs** - GET /ledger/paymentTypeOut
   - Bank transfer
   - Cash
   - Credit card

4. **Module activation** - How to enable travelExpense, project, etc.

5. **Invoice flow** - Exact sequence: order → invoice → payment

6. **Employee employment records** - Required for travel expenses?

7. **Credit note flow** - Does it auto-reverse or need manual posting?

---

## Agentic Fallback Design

For Approach C's agentic mode, the LLM would get these tools:

```python
tools = [
    # Discovery
    {"name": "list_employees", "desc": "GET /employee"},
    {"name": "list_customers", "desc": "GET /customer"},
    {"name": "list_products", "desc": "GET /product"},
    {"name": "list_accounts", "desc": "GET /ledger/account"},
    {"name": "list_vat_types", "desc": "GET /ledger/vatType"},
    {"name": "list_payment_types", "desc": "GET /ledger/paymentTypeOut"},

    # Creation
    {"name": "create_employee", "desc": "POST /employee", "params": {...}},
    {"name": "create_customer", "desc": "POST /customer", "params": {...}},
    {"name": "create_product", "desc": "POST /product", "params": {...}},
    {"name": "create_order", "desc": "POST /order", "params": {...}},
    {"name": "invoice_order", "desc": "PUT /order/{id}/:invoice"},
    {"name": "create_project", "desc": "POST /project"},
    {"name": "create_department", "desc": "POST /department"},
    {"name": "create_travel_expense", "desc": "POST /travelExpense"},

    # Modification
    {"name": "update_employee", "desc": "PUT /employee/{id}"},
    {"name": "register_payment", "desc": "POST /invoice/{id}/:payment"},
    {"name": "create_credit_note", "desc": "POST /invoice/{id}/:createCreditNote"},

    # Deletion
    {"name": "delete_travel_expense", "desc": "DELETE /travelExpense/{id}"},
    {"name": "delete_invoice", "desc": "DELETE /invoice/{id}"},
]
```

The LLM system prompt would include:
- Tripletex API conventions
- Common task patterns
- Norwegian accounting terminology
- Multi-step workflow patterns

---

## Efficiency Optimization Strategy

For perfect scores + efficiency bonus:

### Tier 1 (target: 1 API call each)
- create_employee: 1 POST
- create_customer: 1 POST
- create_product: 1 POST
- create_department: 1 POST

### Tier 2 (target: 2-3 API calls)
- create_invoice: POST order + PUT invoice (2 calls, assumes pre-existing customer/product)
- register_payment: GET invoice + POST payment (2 calls)
- create_project: 1 POST (if manager exists)

### Avoid These Patterns (waste calls + cause 4xx errors):
- Don't GET before POST if you can construct the POST directly
- Don't retry on 4xx - fix the request instead
- Don't list all entities when you know the ID
- Don't use wildcard fields param when you know which fields you need

---

## Language Handling Strategy

The prompt comes in 7 languages. The LLM handles this natively, but we should:

1. **Don't translate** - let the LLM extract fields directly from any language
2. **Preserve Norwegian characters** - ae, oe, aa must stay intact in API calls
3. **Date formats** - Convert any format to YYYY-MM-DD
4. **Currency** - Strip "kr", "NOK", ",-" etc., extract numeric value

### Key Norwegian Accounting Terms
| Norwegian | English | Context |
|-----------|---------|---------|
| ansatt | employee | |
| kunde | customer | |
| produkt | product | |
| faktura | invoice | |
| ordre | order | |
| avdeling | department | |
| prosjekt | project | |
| reiseregning | travel expense | |
| kreditnota | credit note | |
| betaling | payment | |
| mva/merverdiavgift | VAT | |
| regnskap | accounting | |
| bilag/voucher | voucher | |
| hovedbok | general ledger | |
| konto | account | |
| slett | delete | |
| opprett | create | |
| oppdater | update | |
| registrer | register | |

---

## Next Steps Priority

1. **Get sandbox account** - Explore what's pre-populated, discover VAT/payment type IDs
2. **Switch to Approach C** - Hybrid structured + agentic
3. **Submit endpoint** - Start getting real task prompts to learn from
4. **Iterate on failures** - Log all prompts and results, fix patterns
5. **Optimize efficiency** - Once correctness is high, minimize API calls
