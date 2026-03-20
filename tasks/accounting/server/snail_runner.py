"""
Snail-based agent runner. Spawns a sandboxed Claude Code instance
in a Docker container via snail + unleash to handle accounting tasks.

Pattern from pkill-arena:
1. Create tmux session
2. Run unleash <profile> -p '<PROMPT>' inside it
3. Claude Code runs with full permissions and internet access
4. Monitor for completion or timeout
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

SNAIL_PROFILE = "accounting-agent"
TIMEOUT_SECONDS = 240  # Leave 60s buffer from 300s competition timeout


def build_agent_prompt(prompt: str, base_url: str, session_token: str, files: list = None) -> str:
    """Build the prompt that Claude Code will receive inside the container."""
    agent_prompt = f"""Complete this Tripletex accounting task. You have full access to curl, Python, and the internet. Use --dangerously-skip-permissions mode.

## TASK
{prompt}

## TRIPLETEX API ACCESS
Base URL: {base_url}
Auth: Basic auth, username "0", password "{session_token}"

Quick test:
curl -s -u "0:{session_token}" "{base_url}/employee" | python3 -m json.tool

## API CHEAT SHEET
- POST/PUT: JSON body, Content-Type: application/json
- List response: {{"fullResultSize": N, "values": [...]}}
- Create response: {{"value": {{...}}}}
- GET /invoice needs params: invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31
- vatType id=3 for 25% MVA (set on products/order lines)
- POST /ledger/accountingDimensionName for custom dimensions
- POST /ledger/accountingDimensionValue for dimension values
- Voucher postings: amountGross (not amount), row starts at 1
- POST /activity for activities (not /project/id/activity)
- Company needs bank account before invoicing

## WORKFLOW
1. Explore sandbox: curl to list employees, customers, invoices
2. Complete the task with API calls
3. When done, just exit
"""
    if files:
        agent_prompt += "\n## ATTACHED FILES\n"
        for f in files:
            agent_prompt += f"- {f['filename']} ({f['mime_type']})\n"

    return agent_prompt


async def run_in_snail(prompt: str, base_url: str, session_token: str,
                       files: list = None, timeout: int = TIMEOUT_SECONDS) -> dict:
    """Run a task in a sandboxed snail container with Claude Code."""
    start = time.time()
    session_id = f"acc-{int(time.time()) % 100000}"
    session_name = f"{SNAIL_PROFILE}:{session_id}"
    tmux_session = f"acc-agent-{session_id}"

    agent_prompt = build_agent_prompt(prompt, base_url, session_token, files)
    # Escape single quotes for shell
    escaped_prompt = agent_prompt.replace("'", "'\\''")

    try:
        # Step 1: Ensure snail session exists
        log.info(f"Creating snail session: {session_name}")
        create_proc = await asyncio.create_subprocess_exec(
            "snail", "run", session_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Don't wait for it to fully attach - just let it create
        await asyncio.sleep(8)
        create_proc.kill()

        # Step 2: Create tmux session and run unleash inside snail container
        log.info(f"Creating tmux session: {tmux_session}")
        await asyncio.create_subprocess_exec(
            "tmux", "new-session", "-d", "-s", tmux_session, "-n", "agent"
        )
        await asyncio.sleep(1)

        # Step 3: Run unleash with prompt inside the snail container
        # The container name follows snail's naming convention
        container_name = f"snail-slim-{session_id}"
        unleash_cmd = f"docker exec -it {container_name} bash -lic 'unleash claude -p '\"'\"'{escaped_prompt}'\"'\"' --auto --dangerously-skip-permissions'"

        log.info(f"Sending unleash command to tmux session")
        await asyncio.create_subprocess_exec(
            "tmux", "send-keys", "-t", tmux_session, unleash_cmd, "C-m"
        )

        # Step 4: Wait for completion or timeout
        log.info(f"Waiting for agent completion (timeout={timeout}s)")
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            await asyncio.sleep(5)

            # Check if the tmux session is still active
            check = await asyncio.create_subprocess_exec(
                "tmux", "has-session", "-t", tmux_session,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await check.wait()

            if check.returncode != 0:
                # Session ended = agent finished
                break

            # Check if agent process is still running in container
            busy_check = await asyncio.create_subprocess_exec(
                "docker", "exec", container_name,
                "bash", "-c", "pgrep -f 'claude' > /dev/null 2>&1",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await busy_check.wait()
            if busy_check.returncode != 0:
                # Claude process finished
                log.info("Claude process finished in container")
                break

        elapsed = time.time() - start
        log.info(f"Snail agent completed in {elapsed:.1f}s")

        return {
            "elapsed_seconds": round(elapsed, 1),
            "session": session_id,
            "completed": True,
        }

    except Exception as e:
        log.error(f"Snail runner failed: {e}", exc_info=True)
        return {"error": str(e), "completed": False}

    finally:
        # Cleanup
        try:
            subprocess.Popen(
                ["tmux", "kill-session", "-t", tmux_session],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        try:
            subprocess.Popen(
                ["snail", "rm", session_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
