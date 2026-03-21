"""
Task planner: classifies the prompt into a task family and retrieves the relevant playbook.
First LLM call is cheap classification. Playbook is injected into executor context.
"""

import json
import logging
import os
import re
import unicodedata
from pathlib import Path

from openai import OpenAI

log = logging.getLogger(__name__)

PLAYBOOK_DIR = Path(__file__).parent / "playbooks"

# Load all playbooks at startup
PLAYBOOKS = {}
for f in PLAYBOOK_DIR.glob("*.json"):
    pb = json.loads(f.read_text())
    PLAYBOOKS[pb["family"]] = pb

log.info(f"Loaded {len(PLAYBOOKS)} playbooks: {list(PLAYBOOKS.keys())}")

# Prefer more specific families over broad ones like customer when scores tie.
# Invoice must beat timesheet because "faktura med 10 timer" is an invoice, not timesheet.
FAMILY_PRIORITY = {
    "bank_reconciliation": 99,
    "ledger_correction": 98,
    "invoice": 95,
    "travel_expense": 85,
    "salary": 80,
    "project": 75,
    "timesheet": 70,
    "supplier": 60,
    "employee": 50,
    "department": 40,
    "voucher": 30,
    "product": 20,
    "customer": 10,
}


def _compile_keyword_pattern(keyword: str) -> re.Pattern[str]:
    normalized = _normalize_text(keyword)
    tokens = normalized.split()
    if not tokens:
        return re.compile(r"$^")

    # Match inflected forms like "prosjektet", "kunden", and "avdelingar"
    # while keeping a word boundary on the left to avoid noisy substring hits.
    escaped_tokens = [re.escape(token) + r"\w*" for token in tokens]
    pattern = r"(?<!\w)" + r"\s+".join(escaped_tokens)
    return re.compile(pattern, re.IGNORECASE)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.lower()


# Build keyword patterns for fast, boundary-aware matching.
KEYWORD_PATTERNS = {
    family: [
        (_normalize_text(kw), _compile_keyword_pattern(kw))
        for kw in pb.get("keywords", [])
    ]
    for family, pb in PLAYBOOKS.items()
}

# Model setup (reuse from agent)
if os.environ.get("OPENROUTER_API_KEY"):
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "z-ai/glm-5")
elif os.environ.get("OPENAI_API_KEY"):
    client = OpenAI()
    PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "gpt-5.4")
else:
    client = None
    PLANNER_MODEL = None

FAMILIES = list(PLAYBOOKS.keys())

CLASSIFY_PROMPT = f"""Classify this accounting task into exactly one family. Return ONLY a JSON object.

Families: {', '.join(FAMILIES)}

Return: {{"family": "<family_name>", "confidence": "high|medium|low", "reasoning": "<brief>"}}

If the task doesn't fit any family well, use the closest match with confidence "low"."""


def classify_by_keywords(prompt: str) -> tuple[str | None, str]:
    """Fast keyword-based classification. Returns (family, confidence)."""
    prompt_lower = _normalize_text(prompt)
    matches = {}

    for family, patterns in KEYWORD_PATTERNS.items():
        score = 0.0
        for kw, pattern in patterns:
            if pattern.search(prompt_lower):
                # Multi-word phrases are more informative than generic single words.
                score += 2.0 if " " in kw else 1.0
                if " " not in kw and len(kw) >= 10:
                    score += 0.5
        if score:
            matches[family] = score

    if matches:
        best = max(
            matches,
            key=lambda family: (matches[family], FAMILY_PRIORITY.get(family, 0)),
        )
        confidence = "high" if matches[best] >= 3 else "medium"
        return best, confidence
    return None, "low"


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


def classify_by_llm(prompt: str) -> tuple[str, str]:
    """LLM-based classification for when keywords don't match."""
    if not client:
        return "invoice", "low"  # fallback

    response = client.chat.completions.create(
        model=PLANNER_MODEL,
        max_tokens=200,
        temperature=0,
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": prompt[:500]},
        ],
    )

    text = _extract_message_text(response.choices[0].message.content).strip()
    if not text:
        log.warning("Planner model returned empty content")
        return "invoice", "low"
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
        family = result.get("family", "invoice")
        confidence = result.get("confidence", "medium")
        if family not in PLAYBOOKS:
            # Find closest match
            for f in PLAYBOOKS:
                if f in family or family in f:
                    family = f
                    break
            else:
                family = "invoice"  # default fallback
                confidence = "low"
        return family, confidence
    except json.JSONDecodeError:
        log.warning(f"Failed to parse classifier response: {text[:100]}")
        return "invoice", "low"


def plan_task(prompt: str) -> dict:
    """Classify task and return plan with relevant playbook."""

    # Try keywords first (free, fast)
    family, confidence = classify_by_keywords(prompt)
    method = "keyword"

    # If low confidence, use LLM
    if not family or confidence == "low":
        family, confidence = classify_by_llm(prompt)
        method = "llm"

    playbook = PLAYBOOKS.get(family, {})

    log.info(f"Planned: family={family} confidence={confidence} method={method}")

    return {
        "family": family,
        "confidence": confidence,
        "method": method,
        "playbook": playbook,
    }
