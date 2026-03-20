"""
LLM-based task parser. Takes a natural-language prompt (in any of 7 languages)
and extracts a structured task description with intent and fields.

Uses OpenAI-compatible API (works with OpenAI, OpenRouter, etc.)
"""

import json
import logging
import os

from openai import OpenAI

log = logging.getLogger(__name__)

# Use OpenRouter if available, else OpenAI directly
if os.environ.get("OPENROUTER_API_KEY"):
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    MODEL = os.environ.get("LLM_MODEL", "z-ai/glm-5")
elif os.environ.get("OPENAI_API_KEY"):
    client = OpenAI()
    MODEL = os.environ.get("LLM_MODEL", "gpt-5.4")
else:
    raise RuntimeError("No LLM API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

log.info(f"Parser using model: {MODEL}")


def _extract_message_text(content) -> str:
    """Handle providers that return null or structured content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content)

SYSTEM_PROMPT = """You are an expert accounting task parser for the Tripletex accounting system.
You receive a natural-language task description (possibly in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French).
Your job is to extract a structured JSON task description.

Return ONLY valid JSON with this structure:
{
  "task_type": "<one of: create_employee, update_employee, create_customer, update_customer, create_product, create_invoice, register_payment, create_credit_note, create_project, create_department, create_travel_expense, delete_travel_expense, create_order, enable_module, delete_invoice, other>",
  "fields": {
    // All relevant fields extracted from the prompt.
    // Use English field names matching Tripletex API conventions:
    // firstName, lastName, email, phoneNumber, name, address, etc.
    // Include ALL details mentioned in the prompt.
  },
  "steps": [
    // For multi-step tasks, list the steps in order.
    // Each step: {"action": "create_customer", "fields": {...}}
    // For single-step tasks, this can be omitted.
  ],
  "notes": "Any additional context or ambiguity"
}

Key Tripletex field mappings:
- Employee: firstName, lastName, email, phoneNumberMobile, dateOfBirth, employments[].startDate
- Customer: name, email, phoneNumber, postalAddress.addressLine1, postalAddress.postalCode, postalAddress.city
- Product: name, number, priceExcludingVat, priceIncludingVat, vatType.id
- Invoice: customer.id, invoiceDate, dueDate, orders[].orderLines[].product, orders[].orderLines[].count
- Project: name, number, projectManager.id, customer.id, startDate, endDate
- Department: name, departmentNumber
- TravelExpense: employee.id, project.id, department.id, travelDetails

IMPORTANT:
- Extract ALL data from the prompt, including names, numbers, dates, amounts, email addresses.
- For Norwegian names with special characters (ae, oe, aa), preserve them exactly.
- Dates should be in YYYY-MM-DD format.
- Amounts should be numbers (no currency symbols).
- If the task involves multiple steps (e.g., create customer THEN create invoice for that customer), list them in the steps array.
"""


async def parse_task(prompt: str, files: list = None) -> dict:
    """Parse a natural-language prompt into a structured task using an LLM."""
    log.info(f"Parsing prompt: {prompt[:100]}...")

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    text = _extract_message_text(response.choices[0].message.content).strip()
    if not text:
        log.error("Parser model returned empty content")
        task = {"task_type": "other", "fields": {}, "raw_response": "", "original_prompt": prompt}
        if files:
            task["files"] = [{"filename": f.filename, "mime_type": f.mime_type} for f in files]
        return task

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        task = json.loads(text)
    except json.JSONDecodeError:
        log.error(f"Failed to parse LLM response as JSON: {text[:200]}")
        task = {"task_type": "other", "fields": {}, "raw_response": text, "original_prompt": prompt}

    task["original_prompt"] = prompt
    if files:
        task["files"] = [{"filename": f.filename, "mime_type": f.mime_type} for f in files]

    return task
