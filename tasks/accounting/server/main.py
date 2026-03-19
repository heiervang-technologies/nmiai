"""
Tripletex AI Accounting Agent - /solve endpoint for NM i AI 2026.

Receives natural-language accounting task prompts, parses them with an LLM,
executes the corresponding Tripletex API calls, and returns completion status.
"""

import logging
import os

from fastapi import FastAPI, Request
from pydantic import BaseModel

from tripletex_client import TripletexClient
from parser import parse_task
from executor import execute_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent", version="0.1.0")

EXPECTED_API_KEY = os.environ.get("AGENT_API_KEY")


class FileAttachment(BaseModel):
    filename: str
    content_base64: str
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


@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest, request: Request):
    # Optional API key check
    if EXPECTED_API_KEY:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {EXPECTED_API_KEY}":
            log.warning("Invalid API key")

    log.info(f"Received task: {req.prompt[:120]}...")

    # Build Tripletex client
    client = TripletexClient(
        base_url=req.tripletex_credentials.base_url,
        session_token=req.tripletex_credentials.session_token,
    )

    # Parse the prompt into a structured task
    task = await parse_task(req.prompt, req.files)
    log.info(f"Parsed task: {task}")

    # Execute the task against Tripletex API
    result = await execute_task(client, task)
    log.info(f"Execution result: {result}")

    return SolveResponse(status="completed")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
