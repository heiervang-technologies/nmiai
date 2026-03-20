"""
Task planner: classifies the prompt into a task family and retrieves the relevant playbook.
First LLM call is cheap classification. Playbook is injected into executor context.
"""

import json
import logging
import os
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

# Build keyword index for fast matching
KEYWORD_INDEX = {}
for family, pb in PLAYBOOKS.items():
    for kw in pb.get("keywords", []):
        KEYWORD_INDEX[kw.lower()] = family

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
    prompt_lower = prompt.lower()
    matches = {}
    for kw, family in KEYWORD_INDEX.items():
        if kw in prompt_lower:
            matches[family] = matches.get(family, 0) + 1

    if matches:
        best = max(matches, key=matches.get)
        confidence = "high" if matches[best] >= 2 else "medium"
        return best, confidence
    return None, "low"


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

    text = response.choices[0].message.content.strip()
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
