import asyncio
from playwright.async_api import async_playwright
import json

async def main():
    with open("tasks/astar-island/.token") as f:
        token = f.read().strip()
    with open("tasks/astar-island/round1_details.json") as f:
        r_id = json.load(f)["id"]
        
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        
        # Set auth cookie
        await context.add_cookies([{
            "name": "access_token",
            "value": token,
            "domain": "app.ainm.no",
            "path": "/",
        }])
        
        page = await context.new_page()
        
        # Intercept and log all responses
        async def handle_response(response):
            try:
                url = response.url
                if "api.ainm.no" in url or "/api/" in url or "replay" in url or "simulate" in url:
                    print(f"Captured response from: {url}")
                    # Try to parse json to see if it has frames
                    body = await response.json()
                    if isinstance(body, dict):
                        keys = list(body.keys())
                        print(f"  -> JSON keys: {keys}")
                        if "history" in body or "frames" in body or "iterations" in body:
                            print("  -> FOUND POTENTIAL ANIMATION DATA!")
                    elif isinstance(body, list) and len(body) > 0 and isinstance(body[0], dict):
                        print(f"  -> List of dicts, keys of first: {list(body[0].keys())}")
                        if "grid" in body[0] or "step" in body[0] or "time" in body[0]:
                            print("  -> FOUND POTENTIAL ANIMATION DATA LIST!")
            except Exception as e:
                pass

        page.on("response", handle_response)
        
        # Intercept and log all requests
        async def handle_request(request):
            url = request.url
            if "api.ainm.no/astar-island" in url:
                print(f"REQUEST [{request.method}] {url}")
                if request.post_data:
                    print(f"  Payload: {request.post_data}")

        page.on("request", handle_request)
        
        print("Navigating to replay page...")
        await page.goto(f"https://app.ainm.no/submit/astar-island/replay?round={r_id}")
        print("Waiting for page load and network idle...")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
