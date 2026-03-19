# AI Accounting Agent Challenge - NM i AI 2026

## Challenge Overview

Build an AI agent that completes accounting tasks in **Tripletex** (Norwegian cloud accounting software). The agent receives natural-language task prompts, executes Tripletex API calls, and gets scored on correctness and efficiency.

- **Dashboard**: https://app.ainm.no/submit/tripletex
- **Task Docs**: https://app.ainm.no/docs/tripletex/overview
- **Tripletex API Docs**: https://tripletex.no/v2-docs (Swagger/OpenAPI)
- **Tripletex API GitHub**: https://github.com/Tripletex/tripletex-api2

## How It Works

1. Submit an HTTPS `/solve` endpoint URL to the platform
2. Platform provisions a **fresh Tripletex sandbox** for each submission
3. POST request sent to your endpoint with task prompt + credentials
4. Your agent parses the prompt, makes Tripletex API calls via proxy
5. Return `{"status": "completed"}` when done
6. Platform verifies results field-by-field and assigns score

## /solve Endpoint Specification

### Request (POST, JSON)

```json
{
  "prompt": "Opprett en ansatt med navn Ola Nordmann, e-post ola@example.com ...",
  "tripletex_credentials": {
    "base_url": "https://<proxy-url>/v2",
    "session_token": "abc123..."
  },
  "files": [
    {
      "filename": "receipt.pdf",
      "content_base64": "JVBERi0xLjQ...",
      "mime_type": "application/pdf"
    }
  ]
}
```

### Response

```json
{"status": "completed"}
```

- Must respond with HTTP 200 within **300 seconds** (5 minutes)
- HTTPS endpoint required

### Authentication to Tripletex API

**Basic Auth**:
- Username: `0` (the digit zero)
- Password: `session_token` from the request

```python
import requests
requests.get(f"{base_url}/employee", auth=("0", session_token))
```

### Optional API Key

You can set an API key when submitting your endpoint. Platform sends it as Bearer token.

## Task Categories (30 tasks total)

### Tier 1 (x1 multiplier) - Simple operations
- **Employees**: Create employee, update contact info, assign roles
- **Customers**: Register new customers
- **Products**: Create product listings
- **Departments**: Create departments, enable modules

### Tier 2 (x2 multiplier) - Medium complexity
- **Invoicing**: Create invoices, register payments
- **Travel Expenses**: Register expense reports
- **Projects**: Create projects linked to customers

### Tier 3 (x3 multiplier) - Complex multi-step workflows
- **Credit Notes**: Issue credit notes on invoices
- **Corrections**: Delete or reverse entries
- **Multi-step**: Operations requiring prerequisite creation

## Scoring

### Correctness (0.0 - 1.0)
- Field-by-field verification against expected values
- `correctness = points_earned / max_points`
- Example: "Create employee" = 10 points across 5 checks (found, first name, last name, email, role)

### Tier Multiplier
- `base_score = correctness × tier_multiplier`
- Tier 1: max 1.0, Tier 2: max 2.0, Tier 3: max 3.0

### Efficiency Bonus (perfect submissions only)
- Awarded when correctness = 1.0
- Factors: fewer API calls vs best known, fewer 4xx errors
- Can **double** the tier score (max 6.0 for perfect Tier 3)
- Recalculated every 12 hours

### Best Score Policy
- Your score per task = all-time best
- Bad runs never lower your score
- Total score = sum of best scores across all 30 tasks

## Key API Endpoints (via proxy base_url)

| Endpoint | Purpose |
|----------|---------|
| `/employee` | Create/manage employees |
| `/customer` | Create/manage customers |
| `/product` | Create/manage products |
| `/invoice` | Create/manage invoices |
| `/order` | Create/manage orders |
| `/travelExpense` | Travel expense reports |
| `/project` | Project management |
| `/department` | Department management |
| `/ledger/account` | Chart of accounts |
| `/ledger/posting` | Ledger postings |
| `/ledger/voucher` | Vouchers |

### API Tips
- Use `?fields=id,firstName,lastName,*` for field selection
- Pagination: `?from=0&count=100`
- POST/PUT accept JSON body
- DELETE uses ID in URL path
- List responses: `{"fullResultSize": N, "values": [...]}`
- Handle UTF-8 Norwegian characters (ae, oe, aa)

## Task Variants
- **56 variants per task**: 7 languages x 8 datasets
- **Languages**: Norwegian (Bokmal), Nynorsk, English, Spanish, Portuguese, German, French
- Agent must handle multilingual prompts

## Constraints
- Fresh sandbox per submission (no persistent state)
- All API calls through provided proxy base_url
- Concurrent submissions: 10
- Daily submissions: unlimited

## Implementation Strategy

### Phase 1: Basic Infrastructure
1. Create FastAPI/Flask HTTPS server with `/solve` endpoint
2. Parse incoming requests (prompt, credentials, files)
3. Set up Tripletex API client with Basic Auth
4. Deploy with HTTPS (ngrok for dev, cloud for prod)

### Phase 2: Task Parsing
1. Use LLM to parse natural-language prompts into structured task descriptions
2. Identify task type (employee, customer, invoice, etc.)
3. Extract entity fields (names, emails, amounts, dates, etc.)
4. Handle 7 languages - use LLM for translation/understanding

### Phase 3: Task Execution
1. Map task types to Tripletex API calls
2. Handle prerequisite chains (e.g., create customer before invoice)
3. Implement each task category's API workflow
4. Minimize API calls for efficiency bonus

### Phase 4: Optimization
1. Reduce unnecessary GET requests
2. Avoid 4xx errors (validate before calling)
3. Optimize call sequences per task type
4. Test across all task categories

## Tech Stack Recommendation
- **Python + FastAPI**: Quick to build, good async support
- **httpx**: Async HTTP client for Tripletex API calls
- **Claude API**: Parse multilingual prompts into structured data
- **ngrok**: HTTPS tunnel for development
- **Cloud deploy**: Railway/Fly.io/Cloud Run for production

## File Structure
```
tasks/accounting/
  README.md          # This file
  server/
    main.py          # FastAPI /solve endpoint
    tripletex.py     # Tripletex API client
    parser.py        # Prompt parsing with LLM
    tasks/           # Task-specific handlers
      employee.py
      customer.py
      invoice.py
      ...
```
