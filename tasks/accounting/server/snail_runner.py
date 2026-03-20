"""
Snail-based agent runner. Spawns a sandboxed Claude Code instance
in a Docker container to handle complex accounting tasks.

Flow:
1. Receive task prompt + Tripletex credentials
2. Spawn snail container with Claude Code
3. Pass prompt as initial message
4. Claude Code runs with full permissions (curl, python, etc.)
5. Wait for completion or timeout
6. Return result
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

SNAIL_PROFILE = "accounting-agent"
TIMEOUT_SECONDS = 240  # Leave 60s buffer from 300s competition timeout


def build_agent_prompt(prompt: str, base_url: str, session_token: str, files: list = None) -> str:
    """Build the prompt that Claude Code will receive inside the container."""
    agent_prompt = f"""You are completing an accounting task in Tripletex. You have full access to curl, Python, and the internet.

## TASK
{prompt}

## TRIPLETEX API CREDENTIALS
- Base URL: {base_url}
- Authentication: Basic Auth with username "0" and password "{session_token}"

Example curl:
```bash
curl -s -u "0:{session_token}" "{base_url}/employee" | python3 -m json.tool
```

Example Python:
```python
import requests
auth = ("0", "{session_token}")
r = requests.get("{base_url}/employee", auth=auth)
print(r.json())
```

## KEY API FACTS
- POST/PUT use JSON body, Content-Type: application/json
- List responses: {{"fullResultSize": N, "values": [...]}}
- Create responses: {{"value": {{...entity with id...}}}}
- GET /invoice REQUIRES params: invoiceDateFrom, invoiceDateTo
- vatType must be set on products/order lines (id=3 for 25% standard MVA)
- Company needs bank account registered before creating invoices
- POST /ledger/accountingDimensionName for custom dimensions
- POST /ledger/accountingDimensionValue for dimension values
- POST /activity for activities (NOT /project/id/activity)
- Voucher postings: use amountGross, row starts at 1

## INSTRUCTIONS
1. First explore what exists: list employees, customers, invoices, etc.
2. Then complete the task using API calls
3. When done, print "TASK_COMPLETED" on its own line
4. If you encounter errors, read them carefully and adjust your approach
"""

    if files:
        agent_prompt += "\n## ATTACHED FILES\n"
        for f in files:
            agent_prompt += f"- {f['filename']} ({f['mime_type']}): base64 content available\n"

    return agent_prompt


async def run_in_snail(prompt: str, base_url: str, session_token: str,
                       files: list = None, timeout: int = TIMEOUT_SECONDS) -> dict:
    """Run a task in a sandboxed snail container with Claude Code."""
    start = time.time()
    session_name = f"acc-{int(time.time()) % 10000}"

    agent_prompt = build_agent_prompt(prompt, base_url, session_token, files)

    # Write prompt to temp file
    prompt_file = Path(tempfile.mktemp(suffix=".md", prefix="acc-prompt-"))
    prompt_file.write_text(agent_prompt)

    try:
        # Spawn snail session
        log.info(f"Spawning snail session: {SNAIL_PROFILE}:{session_name}")

        # Use unleash inside the container with the prompt
        # The snail session gets CLAUDE_CODE_OAUTH_TOKEN from env file
        proc = await asyncio.create_subprocess_exec(
            "snail", "run", f"{SNAIL_PROFILE}:{session_name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for container to start
        await asyncio.sleep(5)

        # Send the prompt to the container via docker exec
        container_name = f"snail-{SNAIL_PROFILE}-{session_name}"
        docker_cmd = [
            "docker", "exec", container_name,
            "claude", "--print", "--dangerously-skip-permissions",
            "-m", agent_prompt[:50000],  # Truncate if too long
        ]

        log.info(f"Executing claude in container: {container_name}")
        result_proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                result_proc.communicate(),
                timeout=timeout
            )
            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")

            elapsed = time.time() - start
            log.info(f"Snail agent completed in {elapsed:.1f}s")

            return {
                "output": output[-2000:],  # Last 2000 chars
                "error": error[-500:] if error else None,
                "elapsed_seconds": round(elapsed, 1),
                "session": session_name,
                "completed": "TASK_COMPLETED" in output,
            }

        except asyncio.TimeoutError:
            log.warning(f"Snail agent timed out after {timeout}s")
            result_proc.kill()
            return {
                "error": "Timeout",
                "elapsed_seconds": timeout,
                "session": session_name,
                "completed": False,
            }

    except Exception as e:
        log.error(f"Snail runner failed: {e}", exc_info=True)
        return {"error": str(e), "completed": False}

    finally:
        # Clean up
        prompt_file.unlink(missing_ok=True)
        # Remove snail session in background
        try:
            subprocess.Popen(
                ["snail", "rm", f"{SNAIL_PROFILE}:{session_name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
