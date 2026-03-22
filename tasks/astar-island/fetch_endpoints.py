import requests
import json
import re

res = requests.get("https://app.ainm.no/submit/astar-island/replay?round=71451d74-be9f-471f-aacd-a41f3b68a9cd")
scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', res.text)
print(f"Found {len(scripts)} scripts")
for src in set(scripts):
    js_url = "https://app.ainm.no" + src if src.startswith("/") else src
    try:
        js_text = requests.get(js_url).text
        # Look for API paths
        matches = re.findall(r'["\'`](/api/[a-zA-Z0-9_/-]+)', js_text)
        for m in set(matches): print(f"Endpoint in {src}: {m}")
        
        matches2 = re.findall(r'["\'`](/astar-island/[a-zA-Z0-9_/-]+)', js_text)
        for m in set(matches2): print(f"Endpoint in {src}: {m}")
        
        # Look for websockets
        ws_matches = re.findall(r'["\'`](wss?://[^"\'`]+)', js_text)
        for m in set(ws_matches): print(f"Websocket in {src}: {m}")
        
        # Look for ANY full API URL
        api_matches = re.findall(r'["\'`](https://api\.ainm\.no/[^"\'`]+)', js_text)
        for m in set(api_matches): print(f"Full API URL in {src}: {m}")

    except Exception as e:
        print("Error fetching", js_url, e)
