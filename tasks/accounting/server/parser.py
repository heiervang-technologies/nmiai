"""
LLM-based task parser. Takes a natural-language prompt (in any of 7 languages)
and extracts a structured task description with intent and fields.
"""

import json
import logging
import os

import anthropic

log = logging.getLogger(__name__)

client = anthropic.Anthropic()

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
    """Parse a natural-language prompt into a structured task using Claude."""
    log.info(f"Parsing prompt: {prompt[:100]}...")

    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    text = response.content[0].text.strip()

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
