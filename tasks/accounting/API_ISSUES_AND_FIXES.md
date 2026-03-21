# Tripletex API: Known Issues and Correct Signatures

Based on regression testing against sandbox `kkpqfuj-amager.tripletex.dev`.

## Auth
- Basic auth: username=`0`, password=session_token (base64 JWT)
- Session token format: `eyJ...` containing `{"tokenId": ..., "token": "uuid"}`

## VAT Types (Sandbox-Verified)

Valid VAT type IDs in sandbox:
```
id=1  pct=25%   Fradrag inngående avgift, høy sats (INPUT tax)
id=3  pct=25%   Utgående avgift, høy sats (OUTPUT tax)
id=5  pct=0%    Ingen utgående avgift (innenfor mva-loven)
id=6  pct=0%    Ingen utgående avgift (utenfor mva-loven)
id=7  pct=0%    Ingen avgiftsbehandling (inntekter)
id=11 pct=15%   Fradrag inngående avgift, middels sats
id=12 pct=12%   Fradrag inngående avgift, lav sats
id=31 pct=15%   Utgående avgift, middels sats
id=32 pct=12%   Utgående avgift, lav sats
```

### VAT on Products
- **CRITICAL**: This sandbox only accepts `vatType.id=6` for products
- IDs 1-5, 7 all return `Ugyldig mva-kode`
- **Fix**: Try resolved VAT once, fall back to omitting `vatType` entirely
- Without vatType, product defaults to id=6 (0% utenfor mva-loven)

### VAT on Invoices/Orders
- Order lines need `vatType: {id: N}` on each line
- Use `id=3` for 25% standard, `id=31` for 15%, `id=32` for 12%
- Common error: `orders.orderLines.vatType.id: Ugyldig mva-kode`

### VAT on Vouchers
- Voucher postings use `vatType: {id: N}`
- For expense postings use inngående (input) types: id=1 (25%), id=11 (15%), id=12 (12%)
- For revenue postings use utgående (output) types: id=3 (25%), id=31 (15%), id=32 (12%)

## Employee Permissions

### The Permission Error
```
employee.id: Brukertilgangen "Det kan fores lonn, reiseregninger, utlegg m.m. pa bruker" er ikke aktivert
```
This blocks: timesheet entries, travel expenses, salary processing.

### Entitlement API (DOES NOT WORK in sandbox)
- `PUT /employee/entitlement/:grantEntitlementsByTemplate` returns **404**
- Templates like `ALL_ENTITLED`, `INTERNAL_ACCOUNTANT`, `STANDARD` all fail
- **No known fix**: The sandbox employee lacks the permission and we cannot grant it via API

### Workaround
- Use the pre-existing employee (ID from `GET /employee`) which may already have permissions
- Creating new employees does NOT grant these permissions automatically
- `userType: "EXTENDED"` gives admin access but NOT the specific salary/travel/timesheet entitlement

## Endpoint-Specific Issues

### POST /employee/employment
- Error: `Request mapping failed`
- Required body: `{employee: {id}, startDate: "YYYY-MM-DD", employmentType: "ORDINARY"}`
- Must include `employmentType` field

### POST /travelExpense
- Requires `travelDetails` block for classification as travel (not employee expense)
- `travelDetails: {departureDate, returnDate, departure, destination}`
- Employee MUST have the salary/travel/expense entitlement (see permissions section)
- Per diem: POST /travelExpense/perDiemCompensation with `rateType: {id}` from `GET /travelExpense/rate`
- Rate lookup: `GET /travelExpense/rate?isValidDomestic=true&count=50`
- `:deliver` endpoint often fails with validation errors
- Travel expense sub-objects: perDiemCompensation, cost, mileageAllowance, accommodationAllowance

### POST /activity
- **MUST** include `activityType` field
- Valid values: `"GENERAL_ACTIVITY"`, `"PROJECT_SPECIFIC_ACTIVITY"`
- Without it: `activityType: Kan ikke vaere null`

### POST /timesheet/entry
- Required: `{employee: {id}, project: {id}, activity: {id}, date, hours}`
- Activity must be linked to project via `/project/projectActivity` first
- Employee must have timesheet entitlement (see permissions section above)

### POST /product
- Body: `{name, priceExcludingVatCurrency, vatType: {id}}`
- DO NOT use `priceExcludingVat` - use `priceExcludingVatCurrency`
- vatType.id=6 is the only valid option in sandbox (see VAT section)

### POST /department
- Error: `departmentNumber: Nummeret er i bruk`
- **Fix**: Always GET /department first to check if number exists

### POST /invoice
- Requires company bankAccountNumber to be set first
- Body wraps orderLines in orders array
- Each orderLine needs `vatType: {id}`

### GET /invoice
- **REQUIRES** `invoiceDateFrom` and `invoiceDateTo` params
- Without them: validation error

### GET /ledger/posting
- **REQUIRES** `dateFrom` and `dateTo` params
- Without them: `422 Validation failed`

### PUT /project/{id}
- `fixedPrice` field does NOT exist on the project object
- Fixed price must be set during creation or via a different mechanism

## API Call Budget Awareness

Competition scoring includes efficiency multiplier. Every wasted API call hurts.
- Avoid retry loops that try multiple VAT types
- Avoid module activation calls (usually 403 in sandbox)
- Pre-check for existing entities before creating (department, customer, employee)
- Use `count=1` for lookups when you only need one result
