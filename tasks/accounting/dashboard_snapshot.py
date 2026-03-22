#!/usr/bin/env python3
"""Snapshot all 20 visible dashboard results with expanded check details.
Matches each to server logs for family/prompt. Copies to clipboard."""

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

CDP_URL = "http://localhost:9222"
LOG_DIR = Path("/tmp/accounting-logs")


def find_matching_log(utc_time: str) -> dict | None:
    logs = sorted(LOG_DIR.glob("*.json"))
    logs = [l for l in logs if l.name != "summary.jsonl"]
    try:
        h, m = int(utc_time.split(":")[0]), int(utc_time.split(":")[1])
    except:
        return None
    for lf in reversed(logs):
        d = json.load(open(lf))
        ts = d.get("timestamp", "")
        if len(ts) >= 15 and ts.startswith(f"20260321_{h:02d}"):
            ts_min = int(ts[11:13])
            if abs(ts_min - m) <= 3:
                fam = (d.get("plan") or {}).get("family", "?")
                calls = d.get("api_stats", {}).get("calls", []) or []
                errs = sum(1 for c in calls if int(c.get("status", 0)) >= 400)
                writes = sum(1 for c in calls if c.get("method") in ("POST", "PUT", "DELETE") and 200 <= int(c.get("status", 0)) < 300)
                err_details = []
                for c in calls:
                    if int(c.get("status", 0)) >= 400:
                        err_details.append(f"{c.get('method')} {c.get('path','?')[:30]}: {str(c.get('error',''))[:80]}")
                return {
                    "timestamp": ts, "family": fam,
                    "prompt": d.get("prompt", "")[:300],
                    "calls": len(calls), "errors": errs, "writes": writes,
                    "result": (d.get("result") or {}).get("final_message", "")[:300],
                    "err_details": err_details[:3],
                }
    return None


async def main():
    from playwright.async_api import async_playwright

    p = await async_playwright().start()
    b = await p.chromium.connect_over_cdp(CDP_URL)

    pg = None
    for ctx in b.contexts:
        for page in ctx.pages:
            if "submit/tripletex" in page.url:
                pg = page
                break
        if pg:
            break
    if not pg:
        print("ERROR: No submit/tripletex tab found in Brave.")
        return

    # Click ALL task buttons to expand checks
    await pg.evaluate(r"""() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const taskBtns = btns.filter(b => /Task \(/.test(b.textContent));
        taskBtns.forEach(b => b.click());
        return taskBtns.length;
    }""")
    await asyncio.sleep(1.5)

    # Grab full text with all checks expanded
    text = await pg.evaluate("() => document.body.innerText")

    # Parse header
    score_m = re.search(r"Total Score\s*\n\s*([\d.]+)", text)
    rank_m = re.search(r"Rank\s*\n\s*#?(\d+)", text)
    subs_m = re.search(r"(\d+)\s*/\s*300\s*daily", text)

    out = []
    out.append("=" * 90)
    out.append(f"DASHBOARD SNAPSHOT | Score: {score_m.group(1) if score_m else '?'} | "
               f"Rank: #{rank_m.group(1) if rank_m else '?'} | "
               f"Subs: {subs_m.group(1) if subs_m else '?'}/300")
    out.append("=" * 90)
    out.append("")

    # Parse results section
    ridx = text.find("Recent Results")
    if ridx < 0:
        out.append("No Recent Results found")
        print("\n".join(out))
        return

    rtext = text[ridx:]

    # Split into task blocks
    blocks = re.split(r"(?=Task \(\d)", rtext)
    blocks = [b for b in blocks if b.startswith("Task (")]

    for i, block in enumerate(blocks):
        header = re.match(
            r"Task \((\d+\.?\d*)/(\d+)\)\s*\n(\d{2}:\d{2} [AP]M) · ([\d.]+s)\s*\n[\d.]+/\d+ \((\d+)%\)",
            block
        )
        if not header:
            continue

        pts, max_pts, cet_time, duration, pct = header.groups()
        pct_int = int(pct)

        checks = re.findall(r"Check (\d+): (passed|failed)", block)
        passed = sum(1 for _, s in checks if s == "passed")
        failed = sum(1 for _, s in checks if s == "failed")

        # CET -> UTC
        try:
            dt = datetime.strptime(cet_time, "%I:%M %p")
            utc_h = dt.hour - 1
            if utc_h < 0: utc_h += 24
            utc_time = f"{utc_h:02d}:{dt.minute:02d}"
            log = find_matching_log(utc_time)
        except:
            log = None

        family = log["family"] if log else "?"
        api = f"{log['calls']}c {log['errors']}e {log['writes']}w" if log else "?"

        tag = " PERFECT" if pct_int == 100 else " **ZERO**" if pct_int == 0 else ""
        checks_str = f"{passed}P/{failed}F" if checks else "---"

        out.append(f"[{i+1:2d}] {pts:>5}/{max_pts:<3} ({pct:>3}%){tag:>8} | {cet_time} {duration:>7} | {family:<20} | checks:{checks_str:<8} | API:{api}")

        if checks and pct_int < 100:
            check_line = " ".join(f"C{n}:{'P' if s=='passed' else 'F'}" for n, s in checks)
            out.append(f"         checks: {check_line}")

        if pct_int < 50 and log:
            out.append(f"         prompt: {log['prompt'][:140]}")
            out.append(f"         result: {log['result'][:140]}")
            if log.get("err_details"):
                out.append(f"         errors: {'; '.join(log['err_details'][:2])}")

    out.append("")
    out.append("=" * 90)

    full = "\n".join(out)
    print(full)

    try:
        proc = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
        proc.communicate(full.encode())
        print("\n[Copied to clipboard]")
    except Exception as e:
        print(f"\n[Clipboard failed: {e}]")


if __name__ == "__main__":
    asyncio.run(main())
