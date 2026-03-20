# Astar Island Quickstart

## Authentication

All endpoints require authentication. Log in at app.ainm.no, then inspect cookies in your browser to grab your `access_token` JWT.

You can authenticate using either a cookie or a Bearer token header:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>import requests
 
BASE = &quot;https://api.ainm.no&quot;
 
# Option 1: Cookie-based auth
session = requests.Session()
session.cookies.set(&quot;access_token&quot;, &quot;YOUR_JWT_TOKEN&quot;)
 
# Option 2: Bearer token auth
session = requests.Session()
session.headers[&quot;Authorization&quot;] = &quot;Bearer YOUR_JWT_TOKEN&quot;</code></pre>
</figure>

## Step 1: Get the Active Round

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>rounds = session.get(f&quot;{BASE}/astar-island/rounds&quot;).json()
active = next((r for r in rounds if r[&quot;status&quot;] == &quot;active&quot;), None)
 
if active:
    round_id = active[&quot;id&quot;]
    print(f&quot;Active round: {active[&#39;round_number&#39;]}&quot;)</code></pre>
</figure>

## Step 2: Get Round Details

Fetch the detail endpoint to get full round info including `seeds_count` and initial states:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>detail = session.get(f&quot;{BASE}/astar-island/rounds/{round_id}&quot;).json()
 
width = detail[&quot;map_width&quot;]      # 40
height = detail[&quot;map_height&quot;]    # 40
seeds = detail[&quot;seeds_count&quot;]    # 5
print(f&quot;Round: {width}x{height}, {seeds} seeds&quot;)
 
for i, state in enumerate(detail[&quot;initial_states&quot;]):
    grid = state[&quot;grid&quot;]           # height x width terrain codes
    settlements = state[&quot;settlements&quot;]  # [{x, y, has_port, alive}, ...]
    print(f&quot;Seed {i}: {len(settlements)} settlements&quot;)</code></pre>
</figure>

## Step 3: Query the Simulator

You have 50 queries per round, shared across all seeds. Each query reveals a 5-15 cell wide viewport:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>result = session.post(f&quot;{BASE}/astar-island/simulate&quot;, json={
    &quot;round_id&quot;: round_id,
    &quot;seed_index&quot;: 0,
    &quot;viewport_x&quot;: 10,
    &quot;viewport_y&quot;: 5,
    &quot;viewport_w&quot;: 15,
    &quot;viewport_h&quot;: 15,
}).json()
 
grid = result[&quot;grid&quot;]                # 15x15 terrain after simulation
settlements = result[&quot;settlements&quot;]  # settlements in viewport with full stats
viewport = result[&quot;viewport&quot;]        # {x, y, w, h}</code></pre>
</figure>

## Step 4: Build and Submit Predictions

For each seed, submit a `height x width x 6` probability tensor. Each cell has 6 values representing the probability of each terrain class (Empty, Settlement, Port, Ruin, Forest, Mountain). They must sum to 1.0:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>import numpy as np
 
for seed_idx in range(seeds):
    prediction = np.full((height, width, 6), 1/6)  # uniform baseline
 
    # TODO: replace with your model&#39;s predictions
    # prediction[y][x] = [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]
 
    resp = session.post(f&quot;{BASE}/astar-island/submit&quot;, json={
        &quot;round_id&quot;: round_id,
        &quot;seed_index&quot;: seed_idx,
        &quot;prediction&quot;: prediction.tolist(),
    })
    print(f&quot;Seed {seed_idx}: {resp.status_code}&quot;)</code></pre>
</figure>

A uniform prediction scores ~1-5. Use your queries to build better predictions.

> **Warning:** Never assign probability 0.0 to any class. If the ground truth has any non-zero probability for a class you marked as zero, KL divergence becomes infinite and your score for that cell is destroyed. Always enforce a minimum floor (e.g., 0.01) and renormalize. See the [scoring docs](/docs/astar-island/scoring.md#common-pitfalls) for details.

## Using the MCP Server

Add the documentation server to Claude Code for AI-assisted development:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp</code></pre>
</figure>
