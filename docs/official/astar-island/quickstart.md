# Astar Island - Quickstart

## Authentication Setup

### Option 1: Cookie Authentication

Log in via the competition platform. The JWT cookie is set automatically in your browser. For scripts, extract the cookie and pass it:

```python
import httpx

BASE_URL = "https://api.ainm.no/astar-island"
cookies = {"session": "your-jwt-cookie-value"}
client = httpx.Client(base_url=BASE_URL, cookies=cookies)
```

### Option 2: Bearer Token

```python
import httpx

BASE_URL = "https://api.ainm.no/astar-island"
headers = {"Authorization": "Bearer your-token-here"}
client = httpx.Client(base_url=BASE_URL, headers=headers)
```

## Get Active Round

```python
response = client.get("/rounds")
rounds = response.json()
active_round = [r for r in rounds if r.get("active")]
print(f"Active round: {active_round}")
```

## Get Round Details

```python
round_id = 1
response = client.get(f"/rounds/{round_id}")
round_info = response.json()
print(f"Round info: {round_info}")
```

## Query the Simulator

Use POST /simulate to observe a region of the island:

```python
query = {
    "round_id": 1,
    "seed_index": 0,
    "viewport_x": 0,
    "viewport_y": 0,
    "viewport_w": 15,
    "viewport_h": 15,
}
response = client.post("/simulate", json=query)
observation = response.json()
print(f"Observed terrain: {observation}")
```

### Budget Management

You have **50 queries per round** shared across all 5 seeds. Plan your viewports carefully:

```python
# Check remaining budget
response = client.get("/budget")
budget = response.json()
print(f"Remaining queries: {budget}")

# Strategy: 10 queries per seed, covering the full 40x40 grid
# with 15x15 viewports (need ~9 to cover, leaving 1 spare per seed)
viewports = [
    (0, 0), (15, 0), (25, 0),
    (0, 15), (15, 15), (25, 15),
    (0, 25), (15, 25), (25, 25),
]
```

## Submit Predictions

Submit a probability tensor of shape `[40][40][6]`:

```python
import numpy as np

# Initialize with uniform distribution (safe baseline)
prediction = np.full((40, 40, 6), 1.0 / 6.0)

# TODO: Replace with your actual predictions based on observations
# Important: Never use 0.0 - use minimum floor of 0.01
prediction = np.clip(prediction, 0.01, None)

# Normalize each cell to sum to 1.0
prediction = prediction / prediction.sum(axis=2, keepdims=True)

submission = {
    "round_id": 1,
    "seed_index": 0,
    "prediction": prediction.tolist(),
}
response = client.post("/submit", json=submission)
print(f"Submission result: {response.json()}")
```

## Full Example

```python
import httpx
import numpy as np

BASE_URL = "https://api.ainm.no/astar-island"
headers = {"Authorization": "Bearer your-token-here"}

with httpx.Client(base_url=BASE_URL, headers=headers) as client:
    # Get active round
    rounds = client.get("/rounds").json()
    round_id = rounds[0]["id"]

    for seed_index in range(5):
        # Observe the island (use ~10 queries per seed)
        observations = []
        for vx in range(0, 40, 15):
            for vy in range(0, 40, 15):
                w = min(15, 40 - vx)
                h = min(15, 40 - vy)
                query = {
                    "round_id": round_id,
                    "seed_index": seed_index,
                    "viewport_x": vx,
                    "viewport_y": vy,
                    "viewport_w": w,
                    "viewport_h": h,
                }
                obs = client.post("/simulate", json=query).json()
                observations.append(obs)

        # Build prediction from observations
        prediction = np.full((40, 40, 6), 1.0 / 6.0)

        # TODO: Use observations to build informed predictions
        # Apply domain knowledge about simulation mechanics

        # Ensure no zeros and normalize
        prediction = np.clip(prediction, 0.01, None)
        prediction = prediction / prediction.sum(axis=2, keepdims=True)

        # Submit
        submission = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction.tolist(),
        }
        result = client.post("/submit", json=submission).json()
        print(f"Seed {seed_index}: {result}")
```

## MCP Server

An MCP server is available for accessing documentation directly from Claude:

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```
