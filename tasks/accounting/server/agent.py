"""
Agentic executor: LLM with Tripletex API tools.
The LLM reasons about what to do and calls API tools directly.
Self-correcting on errors, handles any task type.
"""

import json
import logging
import os
import time

from openai import OpenAI
from tripletex_client import TripletexClient

log = logging.getLogger(__name__)

# Model config
if os.environ.get("OPENROUTER_API_KEY"):
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    MODEL = os.environ.get("LLM_MODEL", "openai/gpt-5.4")
elif os.environ.get("OPENAI_API_KEY"):
    client = OpenAI()
    MODEL = os.environ.get("LLM_MODEL", "gpt-5.4")
else:
    raise RuntimeError("No LLM API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

log.info(f"Agent using model: {MODEL}")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tripletex_get",
            "description": "GET request to Tripletex API. Use for listing/searching entities. Returns {fullResultSize, values:[...]} for lists or {value:{...}} for single entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path, e.g. /employee, /customer, /ledger/vatType, /employee/123"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters, e.g. {\"firstName\": \"Ola\", \"count\": 10, \"fields\": \"id,name\"}"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tripletex_post",
            "description": "POST request to create a new entity. Returns {value:{...created entity...}}. The body parameter is the JSON request body — do NOT put fields in the path as query parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path only, no query params. E.g. /employee, /customer, /invoice"
                    },
                    "body": {
                        "type": "object",
                        "description": "REQUIRED JSON request body. E.g. {\"firstName\":\"Ola\",\"lastName\":\"Nordmann\",\"email\":\"ola@test.no\"}"
                    }
                },
                "required": ["path", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tripletex_put",
            "description": "PUT request to update an entity or trigger an action. For actions like /:invoice, /:payment, /:createCreditNote, use query params.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path, e.g. /employee/123, /order/456/:invoice"
                    },
                    "body": {
                        "type": "object",
                        "description": "JSON body (optional for action endpoints)"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters, e.g. {\"invoiceDate\": \"2026-03-20\"}"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tripletex_delete",
            "description": "DELETE request to remove an entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path with ID, e.g. /travelExpense/789"
                    }
                },
                "required": ["path"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an expert Tripletex accounting agent. You receive accounting tasks in natural language (Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French) and complete them by calling the Tripletex API.

IMPORTANT: Use the tools correctly. For tripletex_post, put data in the "body" parameter as a JSON object. NEVER put fields as query parameters in the path. The path should be clean like "/employee", not "/employee?firstName=X".

## SANDBOX STATE
Each task gets a FRESH sandbox. However, for some tasks the sandbox comes PRE-POPULATED with relevant data (e.g., a credit note task will have an existing invoice). Check what exists before creating new entities.

Default sandbox contents:
- 1 employee (the account owner)
- 1 department ("Avdeling")
- Full Norwegian chart of accounts
- Active modules: WAGE, ELECTRONIC_VOUCHERS, TIME_TRACKING, API_V2

## TOOL USAGE EXAMPLES

### Creating an employee (correct):
tripletex_get(path="/department")  → get department ID
tripletex_post(path="/employee", body={"firstName":"Ola", "lastName":"Nordmann", "email":"ola@test.no", "userType":"STANDARD", "department":{"id":842220}})

### Searching invoices (correct — date params REQUIRED):
tripletex_get(path="/invoice", params={"invoiceDateFrom":"2020-01-01", "invoiceDateTo":"2030-12-31"})

### Creating a customer (correct):
tripletex_post(path="/customer", body={"name":"Firma AS", "email":"post@firma.no", "organizationNumber":"123456789"})

### WRONG — never do this:
tripletex_post(path="/employee?firstName=Ola&lastName=Nordmann")  ← WRONG! Use body parameter!

## KEY API PATTERNS

### Employee
POST /employee — REQUIRED body fields: {firstName, lastName, userType, department:{id}}
  userType: "STANDARD" (normal), "EXTENDED" (admin access), "NO_ACCESS"
  department: GET /department first to find the ID
  Optional: email, dateOfBirth, phoneNumberMobile, address:{addressLine1, postalCode, city}
PUT /employee/{id} — update. GET first to get id+version, then PUT full object with changes.
IMPORTANT: If task says "administrator" or "admin role", set userType to "EXTENDED"

### Customer
POST /customer — body: {name} is required. Optional: email, phoneNumber, organizationNumber, postalAddress:{addressLine1, postalCode, city}, isPrivateIndividual

### Product
POST /product — body: {name, priceExcludingVatCurrency, vatType:{id:3}}
ALWAYS set vatType. id=3 = 25% standard MVA.

### Invoice (DIRECT — preferred, fewest API calls)
POST /invoice — body: {invoiceDate, invoiceDueDate, customer:{id}, orderLines:[{description, count, unitPriceExcludingVatCurrency, vatType:{id:3}}]}
Creates order + invoice in one call. ALWAYS set vatType on each order line.

### Searching/Listing Invoices
GET /invoice — REQUIRED params: invoiceDateFrom, invoiceDateTo (use wide range like 2020-01-01 to 2030-12-31)

### Payment on Invoice
PUT /invoice/{id}/:payment — params: {paymentDate, paymentTypeId, paidAmount}
To find payment types: GET /invoice/paymentType

### Credit Note
PUT /invoice/{id}/:createCreditNote — params: {date} (today's date YYYY-MM-DD). Optional: comment

### Order → Invoice flow
POST /order — body: {customer:{id}, orderDate, deliveryDate, orderLines:[{product:{id}, count, unitPriceExcludingVatCurrency, vatType:{id:3}}]}
PUT /order/{id}/:invoice — params: {invoiceDate}

### Project
POST /project — body: {name, projectManager:{id}, isInternal:false}. Optional: number, customer:{id}, startDate, endDate
May need module: POST /company/salesmodules body: {name:"PROJECT"}

### Department
POST /department — body: {name, departmentNumber}

### Travel Expense (multi-step)
1. POST /travelExpense — body: {employee:{id}, title, isChargeable:false, isFixedInvoicedAmount:false, isIncludeAttachedReceiptsWhenReinvoicing:false}
2. POST /travelExpense/cost — body: {travelExpense:{id}, date, costCategory:{id}, paymentType:{id}, amountCurrencyIncVat, currency:{id}, vatType:{id}, isPaidByEmployee:true}
3. PUT /travelExpense/:deliver — params: {id}

### Voucher (Journal Entry)
POST /ledger/voucher — body: {date, description, postings:[{account:{id}, amountGross:N, vatType:{id}}]}
Use amountGross, NOT amount.

### Module Activation
POST /company/salesmodules — body: {name:"PROJECT"} or {name:"SMART_PROJECT"}

### Supplier
POST /supplier — body: {name, organizationNumber, email}

## VAT TYPES (known IDs)
- id=3: 25% standard — USE THIS FOR MOST THINGS
- id=31: 15% food
- id=32: 12% transport/hotels
- id=5: 0% within MVA law
- id=6: 0% outside MVA law

## CRITICAL RULES
1. ALWAYS put data in the "body" parameter for POST/PUT, NEVER in the path
2. ALWAYS set vatType on products, order lines, and voucher postings
3. GET /invoice REQUIRES invoiceDateFrom and invoiceDateTo params — use wide range
4. Preserve Norwegian characters exactly (ø, æ, å)
5. Dates in YYYY-MM-DD format, amounts as numbers
6. For updates: GET entity first (need id + version), then PUT full object back
7. Activate modules BEFORE using module-specific features (project, travel expense)
8. Don't make unnecessary API calls — be efficient, avoid trial-and-error
9. Read error messages carefully — they tell you exactly what's wrong
10. Create prerequisites first (customer before invoice, employee before travel expense)
11. Some tasks come with pre-populated data — check what exists before creating

## RESPONSE FORMAT
List: {"fullResultSize": N, "from": 0, "count": N, "values": [...]}
Create/Update: {"value": {...entity with id...}}

Complete the task. When done, stop calling tools and briefly summarize what you did."""


async def run_agent(api_client: TripletexClient, prompt: str, files: list = None) -> dict:
    """Run the agentic tool-use loop."""
    start_time = time.time()

    # Build user message
    user_content = prompt
    if files:
        file_descriptions = "\n".join(
            f"- Attached file: {f['filename']} ({f['mime_type']})"
            for f in files
        )
        user_content += f"\n\nAttached files:\n{file_descriptions}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    max_iterations = 25
    for iteration in range(max_iterations):
        elapsed = time.time() - start_time
        if elapsed > 240:  # Leave 60s buffer from 300s timeout
            log.warning(f"Agent timeout after {elapsed:.0f}s, {iteration} iterations")
            break

        log.info(f"Agent iteration {iteration + 1}, elapsed={elapsed:.1f}s")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message

        # Build message dict for appending
        msg_dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        # If no tool calls, agent is done
        if not msg.tool_calls:
            log.info(f"Agent completed in {iteration + 1} iterations, {elapsed:.1f}s")
            break

        # Execute each tool call
        for tc in msg.tool_calls:
            result = await _execute_tool(api_client, tc)
            # Truncate large results
            result_str = json.dumps(result, ensure_ascii=False, default=str)
            if len(result_str) > 4000:
                result_str = result_str[:4000] + "...(truncated)"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    stats = api_client.get_stats()
    elapsed = time.time() - start_time
    return {
        "iterations": iteration + 1,
        "elapsed_seconds": round(elapsed, 1),
        "api_calls": stats["total_calls"],
        "api_errors": stats["errors_4xx"],
        "final_message": msg.content if msg else None,
    }


async def _execute_tool(api_client: TripletexClient, tool_call) -> dict:
    """Execute a single tool call against the Tripletex API."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON in tool arguments: {tool_call.function.arguments[:200]}"}

    path = args.get("path", "")
    body = args.get("body")
    params = args.get("params")

    try:
        if name == "tripletex_get":
            return await api_client.get(path, params=params)
        elif name == "tripletex_post":
            return await api_client.post(path, json=body)
        elif name == "tripletex_put":
            return await api_client.put(path, json=body, params=params)
        elif name == "tripletex_delete":
            return await api_client.delete(path)
        else:
            return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        error_msg = str(e)
        # Try to extract response body for HTTP errors
        if hasattr(e, 'response'):
            try:
                error_msg = e.response.text[:1000]
            except Exception:
                pass
        log.warning(f"Tool {name} {path} failed: {error_msg[:200]}")
        return {"error": error_msg[:1000]}
