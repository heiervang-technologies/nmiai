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

CORE_PROMPT = """You are an expert Tripletex accounting agent. You complete accounting tasks by calling the Tripletex API via tools.

## UNIVERSAL RULES
1. For tripletex_post/put: data goes in "body" param as JSON. NEVER put fields in the path URL.
2. ALWAYS set vatType on products, order lines, voucher postings (id=3 for 25% standard).
3. GET /invoice REQUIRES params: invoiceDateFrom, invoiceDateTo (use "2020-01-01" to "2030-12-31").
4. Preserve Norwegian characters exactly (ø, æ, å). Dates: YYYY-MM-DD. Amounts: numbers only.
5. For updates: GET first (need id + version), then PUT full object with changes.
6. Read API error messages carefully — they tell you exactly what field is wrong.
7. Some tasks have pre-populated data (customers, invoices). Check what exists first.
8. Create prerequisites before dependents (customer before invoice, employee before project).

## SANDBOX DEFAULT STATE
- 1 employee (account owner), 1 department, full chart of accounts
- Active modules: WAGE, ELECTRONIC_VOUCHERS, TIME_TRACKING, API_V2
- No bank account on company (must register one before creating invoices)

## RESPONSE FORMAT
List: {"fullResultSize": N, "values": [...]}
Create/Update: {"value": {...entity with id...}}

## VAT TYPES
id=3: 25% standard, id=31: 15% food, id=32: 12% transport, id=5/6: 0% exempt

Complete the task efficiently. When done, stop calling tools and briefly summarize what you did."""


def build_system_prompt(playbook: dict | None = None) -> str:
    """Build system prompt: core + injected playbook."""
    parts = [CORE_PROMPT]

    if playbook:
        pb_text = f"\n## TASK-SPECIFIC PLAYBOOK: {playbook.get('family', 'unknown').upper()}\n"
        pb_text += f"Description: {playbook.get('description', '')}\n"

        if playbook.get("preflight"):
            pb_text += "\nPreflight steps:\n"
            for step in playbook["preflight"]:
                pb_text += f"- {step}\n"

        if playbook.get("endpoints"):
            pb_text += "\nEndpoints:\n"
            for ep in playbook["endpoints"]:
                pb_text += f"- {ep}\n"

        if playbook.get("common_errors"):
            pb_text += "\nCommon errors to avoid:\n"
            for err in playbook["common_errors"]:
                pb_text += f"- {err}\n"

        if playbook.get("tips"):
            pb_text += f"\nTips: {playbook['tips']}\n"

        parts.append(pb_text)

    return "\n".join(parts)


async def run_agent(api_client: TripletexClient, prompt: str, files: list = None, playbook: dict = None) -> dict:
    """Run the agentic tool-use loop with optional playbook context."""
    start_time = time.time()

    system_prompt = build_system_prompt(playbook)

    # Build user message
    user_content = prompt
    if files:
        file_descriptions = "\n".join(
            f"- Attached file: {f['filename']} ({f['mime_type']})"
            for f in files
        )
        user_content += f"\n\nAttached files:\n{file_descriptions}"

    messages = [
        {"role": "system", "content": system_prompt},
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
