# Tripletex API v2 - Quick Reference for Agent

## Authentication
- Basic Auth: username=`0`, password=`session_token`
- All calls through provided `base_url` proxy

## Response Format
- Single: `{"value": {...entity...}}`
- List: `{"fullResultSize": N, "from": 0, "count": 100, "values": [...]}`
- Use `?fields=id,name,*` to limit response fields
- Paginate with `?from=0&count=1000`

---

## KEY GOTCHAS (read these!)

1. **VAT must be explicit** on products, order lines, voucher postings. Use `GET /ledger/vatType` to get IDs.
2. **Voucher postings use `amountGross`** not `amount`. Docs say "Only gross amounts will be used."
3. **Invoice can include inline orderLines** - no need to create order first!
4. **Delivery address updates** go through `/deliveryAddress/{id}`, NOT parent entity PUT.
5. **Update payloads require IDs** on both main object AND nested objects (address etc).
6. **No required fields in OpenAPI spec** - all marked optional, but creation will fail without key fields.
7. **Travel expense is multi-step**: create → add costs → deliver → approve → create vouchers.
8. **Credit notes** create a new inverse invoice. Original gets `isCredited: true`.
9. **Employee employment** is a sub-resource: `/employee/employment`

---

## Endpoint Quick Reference

### Employee
```
POST /employee                    {firstName, lastName}
PUT  /employee/{id}               Update (include id in body + nested objects)
GET  /employee                    Search (?firstName=X&lastName=Y)
POST /employee/employment         Create employment record
```

### Customer
```
POST /customer                    {name}
PUT  /customer/{id}               Update
GET  /customer                    Search (?name=X)
DELETE /customer/{id}
```
Optional: email, phoneNumber, organizationNumber, postalAddress, invoiceEmail, isPrivateIndividual

### Product
```
POST /product                     {name}
PUT  /product/{id}                Update
GET  /product                     Search
```
Optional: number, priceExcludingVatCurrency, priceIncludingVatCurrency, vatType:{id}

### Invoice (CRITICAL - two methods)
```
# Method 1: Direct with inline order lines (PREFERRED - fewest API calls)
POST /invoice
{
  "invoiceDate": "2026-03-20",
  "invoiceDueDate": "2026-04-20",
  "customer": {"id": 123},
  "orderLines": [
    {"description": "Consulting", "count": 10,
     "unitPriceExcludingVatCurrency": 1500, "vatType": {"id": 3}}
  ]
}

# Method 2: Order first, then convert
POST /order  →  PUT /order/{id}/:invoice?invoiceDate=2026-03-20
```

### Invoice Payment
```
PUT /invoice/{id}/:payment?paymentDate=2026-03-20&paymentTypeId=N&paidAmount=X
```
Get payment types: `GET /invoice/paymentType`

### Credit Note
```
PUT /invoice/{id}/:createCreditNote?date=2026-03-20
```
Optional: comment, creditNoteEmail, sendToCustomer

### Order
```
POST /order                       {customer:{id}, deliveryDate, orderDate}
PUT  /order/{id}/:invoice         Convert to invoice
```

### Project
```
POST /project                     {name, projectManager:{id}, isInternal}
PUT  /project/{id}
GET  /project
```

### Department
```
POST /department                  {name, departmentNumber}
PUT  /department/{id}
GET  /department
```

### Travel Expense
```
POST /travelExpense               {employee:{id}, isChargeable, isFixedInvoicedAmount, isIncludeAttachedReceiptsWhenReinvoicing}
POST /travelExpense/cost          Add cost line
POST /travelExpense/mileageAllowance  Add mileage
POST /travelExpense/perDiemCompensation  Add per diem
PUT  /travelExpense/:deliver?id=X  Submit for approval
PUT  /travelExpense/:approve?id=X  Approve
DELETE /travelExpense/{id}         Delete
```

### Voucher (Journal Entry)
```
POST /ledger/voucher              {date, description, postings:[...]}
PUT  /ledger/voucher/{id}/:reverse  Reverse voucher
DELETE /ledger/voucher/{id}
```
Posting: `{account:{id}, amountGross:N, date:"", vatType:{id}}`

### Ledger Lookups
```
GET /ledger/vatType               VAT types (read-only)
GET /ledger/account               Chart of accounts
GET /ledger/paymentTypeOut        Outgoing payment types
GET /ledger/voucherType           Voucher types
```

### Supplier
```
POST /supplier                    {name}
GET  /supplier
POST /supplierInvoice/{id}/:addPayment?paymentType=N&amount=X&paymentDate=Y
```

### Modules
```
POST /company/salesmodules        Activate module {name: "PROJECT"}
GET  /company/{id}                Get company (includes module flags)
```
Module names: API_V2, WAGE, SMART_WAGE, TIME_TRACKING, PROJECT, SMART_PROJECT, OCR, ELECTRONIC_VOUCHERS, LOGISTICS

### Contact
```
POST /contact                     {firstName, lastName, customer:{id}}
```

---

## Common VAT Types (Norwegian)
| Rate | Description | Likely ID |
|------|-------------|-----------|
| 25% | Standard MVA | 3 |
| 15% | Food (mat) | 5 |
| 12% | Transport | varies |
| 0% | Exempt | 6 |

**Always verify with `GET /ledger/vatType` on the actual sandbox!**

---

## Common Payment Types
Get from `GET /invoice/paymentType` or `GET /ledger/paymentTypeOut`

---

## Norwegian Accounting Terms
| Norwegian | English | API Entity |
|-----------|---------|------------|
| ansatt | employee | /employee |
| kunde | customer | /customer |
| produkt | product | /product |
| faktura | invoice | /invoice |
| ordre | order | /order |
| avdeling | department | /department |
| prosjekt | project | /project |
| reiseregning | travel expense | /travelExpense |
| kreditnota | credit note | /invoice/:createCreditNote |
| betaling | payment | /invoice/:payment |
| mva | VAT | /ledger/vatType |
| bilag | voucher | /ledger/voucher |
| leverandor | supplier | /supplier |
| konto | account | /ledger/account |
| hovedbok | general ledger | /ledger |
| postering | posting | /ledger/posting |
