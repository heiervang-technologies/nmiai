import asyncio
import os
import httpx

async def main():
    token = os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    if not token:
        # try reading from latest log file
        import glob
        import json
        logs = glob.glob("/tmp/accounting-logs/*.json")
        if logs:
            with open(max(logs, key=os.path.getctime)) as f:
                d = json.load(f)
                # Not available directly in logs probably
                print("No token found")
    print("Cannot proceed without a token, let's just inspect the error traces")

if __name__ == "__main__":
    asyncio.run(main())
