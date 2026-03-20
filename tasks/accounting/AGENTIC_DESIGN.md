# Agentic Design: LLM with Tripletex API Tools

## Why Agentic > Structured Parsing

The challenge has **30 tasks x 56 variants** = 1,680 possible prompts. Pre-coding handlers
for every variant is fragile. An agentic approach where the LLM *reasons* about what to do
and calls API tools directly is:

1. **Robust** to prompt variations across 7 languages
2. **Self-correcting** - can read error responses and retry differently
3. **Discoverable** - can explore the API to figure out unknown patterns
4. **Lower dev effort** - one system handles all 30 tasks

The tradeoff is efficiency (more API calls), but we can optimize known patterns.

## Architecture

```
POST /solve
  ↓
  Prompt + Credentials
  ↓
  ┌─────────────────────────────────────────────────┐
  │ LLM Agent Loop                                  │
  │                                                  │
  │  System prompt: Tripletex API expert             │
  │  Tools: Tripletex API operations                 │
  │                                                  │
  │  1. LLM reads prompt                             │
  │  2. LLM decides which tool to call               │
  │  3. Tool executes API call, returns result        │
  │  4. LLM reads result, decides next action         │
  │  5. Repeat until task complete                    │
  │                                                  │
  │  Max iterations: 15                               │
  │  Timeout: 240s (leave 60s buffer)                │
  └─────────────────────────────────────────────────┘
  ↓
  {"status": "completed"}
```

## Tool Definitions

### Read Operations (for discovery)
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "tripletex_get",
            "description": "Make a GET request to any Tripletex API endpoint. Use for listing/searching entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "API path e.g. /employee, /customer, /ledger/vatType"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters e.g. {\"firstName\": \"Ola\", \"count\": 10}"
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
            "description": "Make a POST request to create a new entity in Tripletex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "body": {"type": "object", "description": "JSON body for the request"}
                },
                "required": ["path", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tripletex_put",
            "description": "Make a PUT request to update an entity or trigger an action (e.g. /:invoice).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "body": {"type": "object"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tripletex_delete",
            "description": "Make a DELETE request to remove an entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    }
]
```

## System Prompt for Agent

```
You are an expert Tripletex accounting agent. You receive a task in natural language
(Norwegian, English, or other languages) and must complete it by calling the Tripletex API.

## API Basics
- Base URL is already configured. Just provide the path (e.g., /employee).
- Auth is handled automatically.
- List responses have format: {"fullResultSize": N, "values": [...]}
- POST/PUT return: {"value": {...created/updated entity...}}

## Key Endpoints
- POST /employee - Create employee {firstName, lastName, email, dateOfBirth}
- PUT /employee/{id} - Update employee
- POST /customer - Create customer {name, email, phoneNumber, postalAddress: {addressLine1, postalCode, city}}
- POST /product - Create product {name, number, priceExcludingVat, vatType: {id: N}}
- POST /department - Create department {name, departmentNumber}
- POST /project - Create project {name, number, projectManager: {id}, customer: {id}}
- POST /order - Create order {customer: {id}, orderDate, deliveryDate, orderLines: [{product: {id}, count, unitPriceExcludingVatCurrency}]}
- PUT /order/{id}/:invoice - Convert order to invoice {invoiceDate, invoiceDueDate}
- POST /invoice/{id}/:payment - Register payment {paymentDate, paymentType: {id}, amount}
- POST /invoice/{id}/:createCreditNote - Create credit note
- POST /travelExpense - Create travel expense {employee: {id}, title}
- DELETE /travelExpense/{id} - Delete travel expense
- GET /ledger/vatType - List VAT types (need IDs for products)
- GET /ledger/paymentTypeOut - List payment types (need IDs for payments)

## Common VAT Types (may vary by sandbox)
- Standard 25%: usually id=3
- Reduced 15%: usually id=5
- Zero-rated: usually id=6

## Invoicing Flow
1. Create customer (if needed)
2. Create product (if needed)
3. POST /order with customer and orderLines
4. PUT /order/{orderId}/:invoice to convert to invoice

## Rules
- Extract ALL details from the prompt. Don't skip any fields mentioned.
- Preserve Norwegian characters exactly (ø, æ, å).
- Convert dates to YYYY-MM-DD format.
- Be efficient - don't make unnecessary API calls.
- If a POST fails, read the error and fix the request.
- For multi-step tasks, create prerequisites first (customer before invoice).
```

## Agent Loop Implementation

```python
async def run_agent(client: TripletexClient, prompt: str, files: list = None):
    """Run the agentic loop: LLM + Tripletex API tools."""

    messages = [{"role": "user", "content": prompt}]

    for iteration in range(15):  # Max 15 tool-call rounds
        response = llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message
        messages.append(msg)

        # If no tool calls, agent is done
        if not msg.tool_calls:
            break

        # Execute each tool call
        for tool_call in msg.tool_calls:
            result = await execute_tool(client, tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return {"iterations": iteration + 1}


async def execute_tool(client, tool_call):
    """Execute a single tool call against Tripletex API."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    try:
        if name == "tripletex_get":
            return await client.get(args["path"], args.get("params"))
        elif name == "tripletex_post":
            return await client.post(args["path"], json=args.get("body"))
        elif name == "tripletex_put":
            return await client.put(args["path"], json=args.get("body"))
        elif name == "tripletex_delete":
            status = await client.delete(args["path"])
            return {"status": status, "deleted": True}
    except httpx.HTTPStatusError as e:
        return {"error": e.response.status_code, "detail": e.response.text}
    except Exception as e:
        return {"error": str(e)}
```

## Optimization: Fast Path for Known Tasks

Before entering the agent loop, try to match the prompt to known task patterns:

```python
FAST_PATTERNS = {
    "create_employee": {
        "keywords": ["opprett", "ansatt", "create", "employee", "erstelle", "Mitarbeiter"],
        "handler": fast_create_employee,
    },
    "create_customer": {
        "keywords": ["kunde", "customer", "registrer", "Kunde"],
        "handler": fast_create_customer,
    },
    # ... etc
}
```

If a fast path matches AND succeeds, we skip the agent loop entirely (1-2 API calls).
If it fails, fall back to the full agent loop.

## Cost & Latency Estimates

| Approach | LLM Calls | API Calls | Latency | Cost/task |
|----------|-----------|-----------|---------|-----------|
| Structured only | 1 | 1-4 | 2-5s | ~$0.01 |
| Agentic only | 1-5 | 3-10 | 5-20s | ~$0.05-0.20 |
| Hybrid (known) | 1 | 1-4 | 2-5s | ~$0.01 |
| Hybrid (unknown) | 1-5 | 3-10 | 5-20s | ~$0.05-0.20 |

With 300s timeout and unlimited submissions, cost is the main concern.
GPT-5.4 pricing TBD but should be manageable for a competition.

## Error Recovery

The agent can self-correct common errors:

1. **400 Bad Request** → Read error message, fix field names/types, retry
2. **404 Not Found** → Entity doesn't exist, create it first
3. **422 Unprocessable** → Validation error, adjust values
4. **Missing prerequisite** → Create the dependent entity (customer for invoice)

This is the KEY advantage of the agentic approach - structured handlers just crash.
