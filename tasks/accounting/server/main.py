"""
Tripletex AI Accounting Agent - /solve endpoint for NM i AI 2026.

Agentic approach: LLM with Tripletex API tools.
Logs all requests/results for contextual bandit iteration.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel

from tripletex_client import TripletexClient
from agent import run_agent
from planner import plan_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent", version="0.2.0")

EXPECTED_API_KEY = os.environ.get("AGENT_API_KEY")
LOG_DIR = Path(os.environ.get("LOG_DIR", "/tmp/accounting-logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


class FileAttachment(BaseModel):
    filename: str
    content_base64: str = ""
    mime_type: str


class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    prompt: str
    tripletex_credentials: TripletexCredentials
    files: list[FileAttachment] = []


class SolveResponse(BaseModel):
    status: str = "completed"


def log_request(req: SolveRequest, result: dict, stats: dict, elapsed: float, plan: dict = None):
    """Log every request for bandit-style analysis."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    entry = {
        "timestamp": ts,
        "prompt": req.prompt,
        "files": [{"filename": f.filename, "mime_type": f.mime_type, "content_len": len(f.content_base64 or "")} for f in req.files],
        "base_url": req.tripletex_credentials.base_url,
        "plan": plan,
        "result": result,
        "api_stats": stats,
        "elapsed_seconds": round(elapsed, 1),
    }
    log_file = LOG_DIR / f"{ts}.json"
    log_file.write_text(json.dumps(entry, ensure_ascii=False, indent=2, default=str))
    log.info(f"Logged to {log_file}")

    # Also append to a single summary file for quick review
    summary_file = LOG_DIR / "summary.jsonl"
    summary = {
        "ts": ts,
        "prompt_preview": req.prompt[:150],
        "family": plan.get("family") if plan else None,
        "confidence": plan.get("confidence") if plan else None,
        "api_calls": stats.get("total_calls", 0),
        "api_errors": stats.get("errors_4xx", 0),
        "iterations": result.get("iterations", 0),
        "elapsed": round(elapsed, 1),
    }
    with open(summary_file, "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest, request: Request):
    if EXPECTED_API_KEY:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {EXPECTED_API_KEY}":
            log.warning("Invalid API key")

    log.info(f"=== NEW TASK === {req.prompt[:120]}...")

    client = TripletexClient(
        base_url=req.tripletex_credentials.base_url,
        session_token=req.tripletex_credentials.session_token,
    )

    start = time.time()
    result = {}
    plan = None
    try:
        # Phase 1: Plan — classify task family and retrieve playbook
        plan = plan_task(req.prompt)
        log.info(f"Plan: family={plan['family']} confidence={plan['confidence']} method={plan['method']}")

        # Phase 2: Execute — run agent with playbook context
        files = None
        if req.files:
            files = [
                {"filename": f.filename, "mime_type": f.mime_type, "content_base64": f.content_base64}
                for f in req.files
            ]

        result = await run_agent(client, req.prompt, files, playbook=plan.get("playbook"))
        log.info(f"Agent result: iterations={result.get('iterations')}, "
                 f"api_calls={result.get('api_calls')}, errors={result.get('api_errors')}")
    except Exception as e:
        log.error(f"Agent failed: {e}", exc_info=True)
        result = {"error": str(e)}
    finally:
        stats = client.get_stats()
        elapsed = time.time() - start
        log_request(req, result, stats, elapsed, plan)
        await client.close()

    return SolveResponse(status="completed")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/logs")
async def get_logs():
    """View recent submission logs."""
    summary_file = LOG_DIR / "summary.jsonl"
    if not summary_file.exists():
        return {"logs": []}
    lines = summary_file.read_text().strip().split("\n")
    return {"logs": [json.loads(line) for line in lines[-20:]]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
