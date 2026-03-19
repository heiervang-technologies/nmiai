# Astar Island Challenge - Strategy & Reference

## Challenge Overview

**Task:** Observe a black-box Norse civilization simulator through a limited viewport and predict the final world state as probability distributions.

**Timeline:** March 19, 18:00 CET to March 22, 15:00 CET (69 hours)

## Core Mechanics

### Map
- **Size:** 40x40 grid
- **Procedurally generated** with map seeds
- Features: ocean borders, fjords, mountain chains, forest patches, initial settlements

### Simulation
- Simulates 50 years of Norse civilization development
- **5 phases per tick:** Growth (settlements expand), Conflict (raids), Trade (wealth exchange), Winter (starvation/collapse), Environment (ruins reclaimed or overtaken)
- 5 seeds per round, each producing a different simulation outcome

### Budget
- **50 queries total per round** (shared across all 5 seeds)
- That's ~10 queries per seed on average
- **Viewport:** 5-15 cells wide/tall (max 15x15)
- **Rate limit:** 5 requests/second

## Terrain / Cell Types

### Initial Map Values (from rounds endpoint)
| Code | Type |
|------|------|
| 0 | Empty |
| 1 | Settlement |
| 2 | Port |
| 3 | Ruin |
| 4 | Forest |
| 5 | Mountain |
| 10 | Ocean |
| 11 | Plains |

### Prediction Classes (6 classes, indices 0-5)
| Index | Class | Color (Hex) | Color (RGB) |
|-------|-------|-------------|-------------|
| 0 | Empty (Ocean+Plains+Empty) | #c8b88a | (200,184,138) |
| 1 | Settlement | #d4760a | (212,118,10) |
| 2 | Port | #0e7490 | (14,116,144) |
| 3 | Ruin | #7f1d1d | (127,29,29) |
| 4 | Forest | #2d5a27 | (45,90,39) |
| 5 | Mountain | #6b7280 | (107,114,128) |

**Note:** Ocean (10) and Plains (11) map to prediction class 0 (Empty). Mountain is static and excluded from scoring.

## API Reference

**Base URL:** `https://api.ainm.no`

**Auth:** Cookie (`access_token` JWT) or Bearer token header.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/astar-island/rounds` | List all rounds |
| GET | `/astar-island/rounds/{round_id}` | Round details + initial states |
| GET | `/astar-island/budget` | Query budget for active round |
| POST | `/astar-island/simulate` | Query simulator (costs 1 query) |
| POST | `/astar-island/submit` | Submit prediction tensor |
| GET | `/astar-island/my-rounds` | Team rounds with scores |
| GET | `/astar-island/my-predictions/{round_id}` | Submitted predictions |
| GET | `/astar-island/analysis/{round_id}/{seed_index}` | Ground truth comparison (post-round) |
| GET | `/astar-island/leaderboard` | Public standings |

### Simulate Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,
  "viewport_h": 15
}
```

### Submit Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,
  "prediction": [[[p0,p1,p2,p3,p4,p5], ...], ...]  // H x W x 6
}
```

### Submission Rules
- Each cell's 6 probabilities must sum to 1.0 (±0.01 tolerance)
- All values non-negative
- **NEVER assign 0.0 to any class** (causes infinite KL divergence)
- Minimum floor: 0.01 per class, then renormalize

## Scoring

### KL Divergence (Entropy-Weighted)
- Per-cell: `KL = Σ p_i × log(p_i / q_i)` where p = ground truth, q = prediction
- Static cells (mountains, fixed ocean) are excluded
- High-entropy cells (uncertain outcomes) count more
- Score range: 0-100 (100 = perfect)
- Round score = average across 5 seeds

### Ground Truth
- Generated from hundreds of simulator runs per seed
- Represents probability distributions, not single outcomes

## Strategy

### Query Allocation (50 queries, 5 seeds)
- **Option A:** 10 queries per seed, systematic coverage
- **Option B:** Concentrate on fewer seeds, uniform prior on others
- **Option C:** Adaptive - observe initial state, then focus queries on dynamic regions

### Critical: Stochastic Simulation
**Each simulate call runs a NEW stochastic simulation** and reveals the viewport result. This means:
- Querying the same region multiple times on the same seed gives you **multiple samples** of the outcome distribution
- Trade-off: cover more area (spatial coverage) vs. repeat observations (better distribution estimates on dynamic cells)
- Since ground truth is generated from hundreds of runs, more samples = closer to ground truth

### Optimal Exploration
1. **First pass:** Get initial states from rounds endpoint (FREE - no query cost)
2. **Identify static regions:** Mountains and deep ocean won't change - predict with certainty
3. **Focus queries on dynamic zones:** Near settlements, forest edges, coastlines
4. **Viewport placement:** Use 15x15 viewports for maximum coverage. 50 queries × 15×15 = 11,250 cells observed. Map is 40×40 = 1,600 cells × 5 seeds = 8,000 total cells. So we can cover ~1.4x the total map area if strategic.
5. **Repeat vs. explore:** Repeating queries on the same dynamic region gives empirical frequency counts for the distribution. This may be more valuable than full coverage on high-entropy areas.

### Prediction Strategy
1. **Static cells:** Mountain → [0,0,0,0,0,1], Ocean → [1,0,0,0,0,0] (with small epsilon)
2. **Observed cells:** Use observed final state, but add uncertainty (don't put all probability on one class)
3. **Unobserved cells:** Use priors from observed regions + initial state + simulation dynamics
4. **Always apply floor:** `max(pred, 0.01)` then renormalize

### Key Insights
- Initial states are FREE (from rounds endpoint) - always fetch these first
- Mountains are static - trivial to predict (class 5 with high confidence)
- Ocean borders are static - predict class 0 (Empty) with high confidence
- Settlements grow, conflict creates ruins, trade creates ports, forests get cleared or reclaim ruins
- Focus observation budget on uncertain, dynamic regions

## Code Template

```python
import requests
import numpy as np

BASE = "https://api.ainm.no"
TOKEN = "YOUR_JWT_TOKEN"

session = requests.Session()
session.cookies.set("access_token", TOKEN)

# 1. Get active round
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)
if not active:
    print("No active round")
    exit()

round_id = active["id"]

# 2. Get initial states (FREE)
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
width = detail["map_width"]
height = detail["map_height"]
seeds = detail["seeds_count"]

# 3. Query simulator (costs 1 per call, 50 total budget)
def observe(seed_idx, x, y, w=15, h=15):
    return session.post(f"{BASE}/astar-island/simulate", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "viewport_x": x, "viewport_y": y,
        "viewport_w": w, "viewport_h": h,
    }).json()

# 4. Build predictions
def build_prediction(height, width, observations, initial_grid):
    pred = np.full((height, width, 6), 1/6)  # uniform prior

    # Set static cells from initial state
    for r in range(height):
        for c in range(width):
            cell = initial_grid[r][c]
            if cell == 5:  # Mountain
                pred[r, c] = [0.01]*5 + [0.94]  # high confidence mountain
            elif cell == 10:  # Ocean
                pred[r, c] = [0.94] + [0.01]*5  # high confidence empty

    # Incorporate observations
    for obs in observations:
        # ... process viewport data
        pass

    # Floor and renormalize
    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred

# 5. Submit
def submit_prediction(seed_idx, prediction):
    return session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    }).json()
```

## MCP Server
```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```

## Python Setup (uv)
```bash
cd ~/ht/nmiai
uv init
uv add requests numpy
```

## File Structure
```
tasks/astar-island/
├── README.md          # This file
├── solver.py          # Main solving script
├── explorer.py        # Query strategy & viewport placement
└── analysis.py        # Post-round analysis tools
```
