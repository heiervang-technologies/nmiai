#!/usr/bin/env python3
"""Auto-click submit via raw CDP websocket. No playwright dependency issues."""
import asyncio
import json
import time
import sys
import subprocess
import urllib.request
import websockets

MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 30
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 120
ENDPOINT = "https://newer-ate-sport-upc.trycloudflare.com/solve"

def send_master(msg):
    m = f'<agent id="bayesian-opt" role="score-optimizer" pane="%28">{msg}</agent>'
    subprocess.run(["tmux-tool", "send", "%8", m], capture_output=True)
    time.sleep(0.5)
    subprocess.run(["tmux", "send-keys", "-t", "%8", "Enter"], capture_output=True)
    time.sleep(0.3)
    subprocess.run(["tmux", "send-keys", "-t", "%8", "Enter"], capture_output=True)

async def cdp_call(ws, method, params=None):
    msg_id = int(time.time() * 1000) % 1000000
    msg = {"id": msg_id, "method": method}
    if params:
        msg["params"] = params
    await ws.send(json.dumps(msg))
    while True:
        resp = json.loads(await ws.recv())
        if resp.get("id") == msg_id:
            return resp.get("result", {})

async def main():
    # Find the tripletex tab
    tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
    ws_url = None
    for t in tabs:
        if "submit/tripletex" in t.get("url", ""):
            ws_url = t["webSocketDebuggerUrl"]
            break
    if not ws_url:
        print("ERROR: No tripletex tab found")
        return

    print(f"Connecting to {ws_url[:60]}")
    async with websockets.connect(ws_url, max_size=10_000_000, ping_interval=None, ping_timeout=None) as ws:
        print("Connected!")

        # Fill endpoint URL
        js_fill = f"""
        (function() {{
            const inputs = document.querySelectorAll('input');
            for (const inp of inputs) {{
                if (inp.placeholder && inp.placeholder.includes('solve')) {{
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    setter.call(inp, '{ENDPOINT}');
                    inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return 'filled';
                }}
            }}
            return 'not_found';
        }})()
        """
        result = await cdp_call(ws, "Runtime.evaluate", {"expression": js_fill, "returnByValue": True})
        print(f"Fill: {result.get('result', {}).get('value', '?')}")
        await asyncio.sleep(1)

        for i in range(MAX):
            ts = time.strftime("%H:%M:%S")

            # Click submit button
            js_click = """
            (function() {
                const btns = document.querySelectorAll('button');
                for (const b of btns) {
                    if (b.textContent.trim() === 'Submit' && !b.disabled) {
                        b.click();
                        return 'clicked';
                    } else if (b.textContent.trim() === 'Submit') {
                        return 'disabled';
                    }
                }
                return 'not_found';
            })()
            """
            result = await cdp_call(ws, "Runtime.evaluate", {"expression": js_click, "returnByValue": True})
            status = result.get("result", {}).get("value", "?")
            print(f"[{ts}] #{i+1}/{MAX} {status}", flush=True)

            if status != "clicked":
                await asyncio.sleep(10)
                continue

            # Wait for result (poll for submit button to re-enable, max 5 min)
            for _ in range(60):
                await asyncio.sleep(5)
                js_score = r"""
                (function() {
                    const text = document.body.innerText;
                    const s = text.match(/Total Score\s*\n\s*([\d.]+)/);
                    const r = text.match(/Rank\s*\n\s*#?(\d+)/);
                    const sub = text.match(/(\d+)\s*\/\s*300\s*daily/);
                    const res = text.match(/Recent Results[\s\S]*?Task \((\d+\.?\d*\/\d+)\)/);
                    const submitReady = Array.from(document.querySelectorAll('button')).some(b => b.textContent.trim() === 'Submit' && !b.disabled);
                    return {
                        score: s ? s[1] : '?',
                        rank: r ? r[1] : '?',
                        subs: sub ? sub[1] : '?',
                        latest: res ? res[1] : '?',
                        ready: submitReady
                    };
                })()
                """
                result = await cdp_call(ws, "Runtime.evaluate", {"expression": js_score, "returnByValue": True})
                data = result.get("result", {}).get("value", {})
                if data.get("ready"):
                    report = f"[{i+1}/{MAX}] Score:{data.get('score','?')} Rank:#{data.get('rank','?')} Subs:{data.get('subs','?')}/300 Latest:{data.get('latest','?')}"
                    print(report, flush=True)
                    send_master(report)
                    break

            if i < MAX - 1:
                print(f"Waiting {INTERVAL}s...", flush=True)
                await asyncio.sleep(INTERVAL)

    print("DONE!")

asyncio.run(main())
