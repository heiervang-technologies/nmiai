#!/usr/bin/env python3
"""Auto-submit accounting tasks via Playwright CDP.
Clicks Submit via xpath, waits for result, reports to master-accounting."""

import asyncio
import subprocess
import time
import sys

CDP_URL = "http://localhost:9222"
SUBMIT_XPATH = "xpath=/html/body/div[3]/main/div/div/div/div[5]/form/button"
MASTER_PANE = "%8"
SELF_PANE = "%28"
MAX_SUBS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 65


def send_to_master(msg: str):
    agent_msg = f'<agent id="bayesian-opt" role="score-optimizer" pane="{SELF_PANE}">{msg}</agent>'
    subprocess.run(["tmux-tool", "send", MASTER_PANE, agent_msg], capture_output=True)
    time.sleep(0.5)
    subprocess.run(["tmux", "send-keys", "-t", MASTER_PANE, "Enter"], capture_output=True)
    time.sleep(0.3)
    subprocess.run(["tmux", "send-keys", "-t", MASTER_PANE, "Enter"], capture_output=True)


async def main():
    from playwright.async_api import async_playwright

    p = await async_playwright().start()
    b = await p.chromium.connect_over_cdp(CDP_URL)
    # Find the tripletex submit page across all tabs
    pg = None
    for ctx in b.contexts:
        for page in ctx.pages:
            if "submit/tripletex" in page.url:
                pg = page
                break
        if pg:
            break
    if not pg:
        print("ERROR: No submit/tripletex tab found. Open it in Brave first.")
        return

    print(f"Connected: {pg.url}")

    # Ensure endpoint URL is filled
    ENDPOINT = "https://newer-ate-sport-upc.trycloudflare.com/solve"
    await pg.evaluate(f'''() => {{
        const inputs = document.querySelectorAll('input');
        for (const inp of inputs) {{
            if (inp.placeholder && inp.placeholder.includes('solve')) {{
                const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                setter.call(inp, '{ENDPOINT}');
                inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return true;
            }}
        }}
        return false;
    }}''')
    await asyncio.sleep(1)
    print(f"Endpoint set: {ENDPOINT}")

    # Get initial scores
    text = await pg.evaluate("()=>document.body.innerText")
    import re
    score_m = re.search(r"Total Score\s*\n\s*([\d.]+)", text)
    subs_m = re.search(r"(\d+)\s*/\s*300\s*daily", text)
    init_score = score_m.group(1) if score_m else "?"
    init_subs = subs_m.group(1) if subs_m else "?"
    print(f"Score: {init_score} | Subs: {init_subs}/300")
    send_to_master(f"AUTO-SUBMIT STARTING: score={init_score} subs={init_subs}/300. Running {MAX_SUBS} at 1/min.")

    prev_score = float(init_score) if init_score != "?" else 0

    for i in range(MAX_SUBS):
        ts = time.strftime("%H:%M:%S")

        # Grab results text before clicking
        old_text = await pg.evaluate("()=>document.body.innerText")
        old_results = old_text[old_text.find("Recent Results"):old_text.find("Recent Results")+200] if "Recent Results" in old_text else ""

        # Click submit
        btn = pg.locator(SUBMIT_XPATH)
        if await btn.count() == 0 or not await btn.is_enabled():
            print(f"[{ts}] Submit button not available, reloading...")
            await pg.goto("https://app.ainm.no/submit/tripletex")
            await asyncio.sleep(3)
            continue

        await btn.click()
        print(f"[{ts}] #{i+1}/{MAX_SUBS} submitted")

        # Wait for result (poll until Recent Results changes, max 5 min)
        for _ in range(60):
            await asyncio.sleep(5)
            try:
                new_text = await pg.evaluate("()=>document.body.innerText")
                new_results = new_text[new_text.find("Recent Results"):new_text.find("Recent Results")+200] if "Recent Results" in new_text else ""
                if new_results != old_results and new_results:
                    break
            except:
                pass

        # Extract scores
        try:
            text = await pg.evaluate("()=>document.body.innerText")
            score_m = re.search(r"Total Score\s*\n\s*([\d.]+)", text)
            subs_m = re.search(r"(\d+)\s*/\s*300\s*daily", text)
            rank_m = re.search(r"Rank\s*\n\s*#?(\d+)", text)

            score = score_m.group(1) if score_m else "?"
            subs = subs_m.group(1) if subs_m else "?"
            rank = rank_m.group(1) if rank_m else "?"

            # Get latest result
            if "Recent Results" in text:
                results_section = text[text.find("Recent Results"):]
                lines = [l.strip() for l in results_section.split("\n") if l.strip()]
                latest = " | ".join(lines[1:4])
            else:
                latest = "?"

            # Score delta
            try:
                delta = float(score) - prev_score
                delta_str = f" (+{delta:.1f})" if delta > 0 else f" ({delta:.1f})" if delta < 0 else " (=)"
                prev_score = float(score)
            except:
                delta_str = ""

            report = f"[{i+1}/{MAX_SUBS}] Score:{score}{delta_str} Rank:#{rank} Subs:{subs}/300 | {latest[:100]}"
            print(report)
            send_to_master(report)

        except Exception as e:
            print(f"Error extracting: {e}")
            send_to_master(f"[{i+1}/{MAX_SUBS}] Result extraction error: {e}")

        # Wait before next
        if i < MAX_SUBS - 1:
            print(f"Waiting {INTERVAL}s...")
            await asyncio.sleep(INTERVAL)

    send_to_master(f"AUTO-SUBMIT DONE: {MAX_SUBS} complete. Final score={score} rank=#{rank}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
