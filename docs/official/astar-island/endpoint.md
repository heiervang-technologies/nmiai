# API Endpoint Specification

## Base URL

    https://api.ainm.no/astar-island

All endpoints require authentication. The API accepts either:

- **Cookie:** `access_token` JWT cookie (set automatically when you log in at app.ainm.no)
- **Bearer token:** `Authorization: Bearer <token>` header

Both methods use the same JWT token. Use whichever is more convenient for your setup.

## Endpoints Overview

<div class="table-scroll-wrapper">

| Method | Path | Auth | Description |
|----|----|----|----|
| `GET` | `/astar-island/rounds` | Public | List all rounds |
| `GET` | `/astar-island/rounds/{round_id}` | Public | Round details + initial states |
| `GET` | `/astar-island/budget` | Team | Query budget for active round |
| `POST` | `/astar-island/simulate` | Team | Observe one simulation through viewport |
| `POST` | `/astar-island/submit` | Team | Submit prediction tensor |
| `GET` | `/astar-island/my-rounds` | Team | Rounds with your scores, rank, budget |
| `GET` | `/astar-island/my-predictions/{round_id}` | Team | Your predictions with argmax/confidence |
| `GET` | `/astar-island/analysis/{round_id}/{seed_index}` | Team | Post-round ground truth comparison |
| `GET` | `/astar-island/leaderboard` | Public | Astar Island leaderboard |

</div>

## GET /astar-island/rounds

List all rounds with status and timing.

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>[
  {
    &quot;id&quot;: &quot;uuid&quot;,
    &quot;round_number&quot;: 1,
    &quot;event_date&quot;: &quot;2026-03-19&quot;,
    &quot;status&quot;: &quot;active&quot;,
    &quot;map_width&quot;: 40,
    &quot;map_height&quot;: 40,
    &quot;prediction_window_minutes&quot;: 165,
    &quot;started_at&quot;: &quot;2026-03-19T10:00:00Z&quot;,
    &quot;closes_at&quot;: &quot;2026-03-19T12:45:00Z&quot;,
    &quot;round_weight&quot;: 1,
    &quot;created_at&quot;: &quot;2026-03-19T09:00:00Z&quot;
  }
]</code></pre>
</figure>

### Round Status

<div class="table-scroll-wrapper">

| Status      | Meaning                                 |
|-------------|-----------------------------------------|
| `pending`   | Round created but not yet started       |
| `active`    | Queries and submissions open            |
| `scoring`   | Submissions closed, scoring in progress |
| `completed` | Scores finalized                        |

</div>

## GET /astar-island/rounds/{round_id}

Returns round details including **initial map states** for all seeds. Use this to reconstruct the starting terrain locally.

**Note:** Settlement data in initial states shows only position and port status. Internal stats (population, food, wealth, defense) are not exposed.

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;id&quot;: &quot;uuid&quot;,
  &quot;round_number&quot;: 1,
  &quot;status&quot;: &quot;active&quot;,
  &quot;map_width&quot;: 40,
  &quot;map_height&quot;: 40,
  &quot;seeds_count&quot;: 5,
  &quot;initial_states&quot;: [
    {
      &quot;grid&quot;: [[10, 10, 10, ...], ...],
      &quot;settlements&quot;: [
        {
          &quot;x&quot;: 5, &quot;y&quot;: 12,
          &quot;has_port&quot;: true,
          &quot;alive&quot;: true
        }
      ]
    }
  ]
}</code></pre>
</figure>

### Grid Cell Values

<div class="table-scroll-wrapper">

| Value | Terrain    |
|-------|------------|
| 0     | Empty      |
| 1     | Settlement |
| 2     | Port       |
| 3     | Ruin       |
| 4     | Forest     |
| 5     | Mountain   |
| 10    | Ocean      |
| 11    | Plains     |

</div>

## GET /astar-island/budget

Check your team's remaining query budget for the active round.

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;round_id&quot;: &quot;uuid&quot;,
  &quot;queries_used&quot;: 23,
  &quot;queries_max&quot;: 50,
  &quot;active&quot;: true
}</code></pre>
</figure>

## Rate Limits

<div class="table-scroll-wrapper">

| Endpoint         | Limit                      |
|------------------|----------------------------|
| `POST /simulate` | 5 requests/second per team |
| `POST /submit`   | 2 requests/second per team |

</div>

Exceeding these limits returns `429 Too Many Requests`.

## POST /astar-island/simulate

**This is the core observation endpoint.** Each call runs one stochastic simulation and reveals a viewport window of the result. Costs one query from your budget (50 per round).

![Viewport: full map with highlighted window vs. what gets returned](/docs/astar-island/viewport-demo.png)

### Request

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;round_id&quot;: &quot;uuid-of-active-round&quot;,
  &quot;seed_index&quot;: 3,
  &quot;viewport_x&quot;: 10,
  &quot;viewport_y&quot;: 5,
  &quot;viewport_w&quot;: 15,
  &quot;viewport_h&quot;: 15
}</code></pre>
</figure>

<div class="table-scroll-wrapper">

| Field        | Type       | Description                       |
|--------------|------------|-----------------------------------|
| `round_id`   | string     | UUID of the active round          |
| `seed_index` | int (0–4)  | Which of the 5 seeds to simulate  |
| `viewport_x` | int (\>=0) | Left edge of viewport (default 0) |
| `viewport_y` | int (\>=0) | Top edge of viewport (default 0)  |
| `viewport_w` | int (5–15) | Viewport width (default 15)       |
| `viewport_h` | int (5–15) | Viewport height (default 15)      |

</div>

### Response

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;grid&quot;: [[4, 11, 1, ...], ...],
  &quot;settlements&quot;: [
    {
      &quot;x&quot;: 12, &quot;y&quot;: 7,
      &quot;population&quot;: 2.8,
      &quot;food&quot;: 0.4,
      &quot;wealth&quot;: 0.7,
      &quot;defense&quot;: 0.6,
      &quot;has_port&quot;: true,
      &quot;alive&quot;: true,
      &quot;owner_id&quot;: 3
    }
  ],
  &quot;viewport&quot;: {&quot;x&quot;: 10, &quot;y&quot;: 5, &quot;w&quot;: 15, &quot;h&quot;: 15},
  &quot;width&quot;: 40,
  &quot;height&quot;: 40,
  &quot;queries_used&quot;: 24,
  &quot;queries_max&quot;: 50
}</code></pre>
</figure>

The `grid` contains only the viewport region (viewport_h × viewport_w), not the full map. The `settlements` list includes only settlements within the viewport. The `viewport` object confirms the actual viewport bounds (clamped to map edges). `width` and `height` give the full map dimensions.

Each call uses a different random sim_seed, so you get a different stochastic outcome.

### Error Codes

<div class="table-scroll-wrapper">

| Status | Meaning |
|----|----|
| 400 | Round not active, or invalid seed_index |
| 403 | Not on a team |
| 404 | Round not found |
| 429 | Query budget exhausted (50/50) or rate limit exceeded (max 5 req/sec) |

</div>

## POST /astar-island/submit

Submit your prediction for one seed. You must submit all 5 seeds for a complete score.

### Request

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;round_id&quot;: &quot;uuid-of-active-round&quot;,
  &quot;seed_index&quot;: 3,
  &quot;prediction&quot;: [
    [
      [0.85, 0.05, 0.02, 0.03, 0.03, 0.02],
      [0.10, 0.40, 0.30, 0.10, 0.05, 0.05],
      ...
    ],
    ...
  ]
}</code></pre>
</figure>

<div class="table-scroll-wrapper">

| Field | Type | Description |
|----|----|----|
| `round_id` | string | UUID of the active round |
| `seed_index` | int (0–4) | Which seed this prediction is for |
| `prediction` | float\[\]\[\]\[\] | H×W×6 tensor — probability per cell per class |

</div>

### Prediction Format

The `prediction` is a 3D array: `prediction[y][x][class]`

- Outer dimension: **H** rows (height)
- Middle dimension: **W** columns (width)
- Inner dimension: **6** probabilities (one per class)
- Each cell's 6 probabilities must sum to 1.0 (±0.01 tolerance)
- All probabilities must be non-negative

### Class Indices

<div class="table-scroll-wrapper">

| Index | Class                        |
|-------|------------------------------|
| 0     | Empty (Ocean, Plains, Empty) |
| 1     | Settlement                   |
| 2     | Port                         |
| 3     | Ruin                         |
| 4     | Forest                       |
| 5     | Mountain                     |

</div>

### Response

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;status&quot;: &quot;accepted&quot;,
  &quot;round_id&quot;: &quot;uuid&quot;,
  &quot;seed_index&quot;: 3
}</code></pre>
</figure>

Resubmitting for the same seed overwrites your previous prediction. Only the last submission counts.

### Validation Errors

<div class="table-scroll-wrapper">

| Error | Cause |
|----|----|
| `Expected H rows, got N` | Wrong number of rows |
| `Row Y: expected W cols, got N` | Wrong number of columns |
| `Cell (Y,X): expected 6 probs, got N` | Wrong probability vector length |
| `Cell (Y,X): probs sum to S, expected 1.0` | Probabilities don't sum to 1.0 |
| `Cell (Y,X): negative probability` | Negative value in probability vector |

</div>

## GET /astar-island/my-rounds

Returns all rounds enriched with your team's scores, submission counts, rank, and query budget. This is the team-specific version of `/rounds`.

**Auth required.**

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>[
  {
    &quot;id&quot;: &quot;uuid&quot;,
    &quot;round_number&quot;: 1,
    &quot;event_date&quot;: &quot;2026-03-19&quot;,
    &quot;status&quot;: &quot;completed&quot;,
    &quot;map_width&quot;: 40,
    &quot;map_height&quot;: 40,
    &quot;seeds_count&quot;: 5,
    &quot;round_weight&quot;: 1,
    &quot;started_at&quot;: &quot;2026-03-19T10:00:00+00:00&quot;,
    &quot;closes_at&quot;: &quot;2026-03-19T10:45:00+00:00&quot;,
    &quot;prediction_window_minutes&quot;: 165,
    &quot;round_score&quot;: 72.5,
    &quot;seed_scores&quot;: [80.1, 65.3, 71.9, ...],
    &quot;seeds_submitted&quot;: 5,
    &quot;rank&quot;: 3,
    &quot;total_teams&quot;: 12,
    &quot;queries_used&quot;: 48,
    &quot;queries_max&quot;: 50,
    &quot;initial_grid&quot;: [[10, 10, 10, ...], ...]
  }
]</code></pre>
</figure>

<div class="table-scroll-wrapper">

| Field | Type | Description |
|----|----|----|
| `round_score` | float \| null | Your team's average score across all seeds (null if not scored) |
| `seed_scores` | float\[\] \| null | Per-seed scores (null if not scored) |
| `seeds_submitted` | int | Number of seeds your team has submitted predictions for |
| `rank` | int \| null | Your team's rank for this round (null if not scored) |
| `total_teams` | int \| null | Total teams scored in this round |
| `queries_used` | int | Simulation queries used by your team |
| `queries_max` | int | Maximum queries allowed (default 50) |
| `initial_grid` | int\[\]\[\] | Initial terrain grid for the first seed |

</div>

### Error Codes

<div class="table-scroll-wrapper">

| Status | Meaning       |
|--------|---------------|
| 403    | Not on a team |

</div>

## GET /astar-island/my-predictions/{round_id}

Returns your team's submitted predictions for a given round, with derived argmax and confidence grids for easy visualization.

**Auth required.**

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>[
  {
    &quot;seed_index&quot;: 0,
    &quot;argmax_grid&quot;: [[0, 4, 5, ...], ...],
    &quot;confidence_grid&quot;: [[0.85, 0.72, 0.93, ...], ...],
    &quot;score&quot;: 78.2,
    &quot;submitted_at&quot;: &quot;2026-03-19T10:30:00+00:00&quot;
  }
]</code></pre>
</figure>

<div class="table-scroll-wrapper">

| Field | Type | Description |
|----|----|----|
| `seed_index` | int | Which seed this prediction is for (0–4) |
| `argmax_grid` | int\[\]\[\] | H×W grid of predicted class indices (argmax of probability vector) |
| `confidence_grid` | float\[\]\[\] | H×W grid of confidence values (max probability per cell, rounded to 3 decimals) |
| `score` | float \| null | Score for this seed (null if not yet scored) |
| `submitted_at` | string \| null | ISO 8601 timestamp of submission |

</div>

The `argmax_grid` uses the same class indices as the prediction format (0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain).

### Error Codes

<div class="table-scroll-wrapper">

| Status | Meaning       |
|--------|---------------|
| 403    | Not on a team |

</div>

## GET /astar-island/analysis/{round_id}/{seed_index}

Post-round analysis endpoint. Returns your prediction alongside the ground truth for a specific seed, enabling detailed comparison. Only available after a round is completed (or during scoring).

**Auth required.**

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;prediction&quot;: [[[0.85, 0.05, 0.02, 0.03, 0.03, 0.02], ...], ...],
  &quot;ground_truth&quot;: [[[0.90, 0.03, 0.01, 0.02, 0.02, 0.02], ...], ...],
  &quot;score&quot;: 78.2,
  &quot;width&quot;: 40,
  &quot;height&quot;: 40,
  &quot;initial_grid&quot;: [[10, 10, 10, ...], ...]
}</code></pre>
</figure>

<div class="table-scroll-wrapper">

| Field | Type | Description |
|----|----|----|
| `prediction` | float\[\]\[\]\[\] | Your submitted H×W×6 probability tensor |
| `ground_truth` | float\[\]\[\]\[\] | The actual H×W×6 probability distribution (computed from Monte Carlo simulations) |
| `score` | float \| null | Your score for this seed |
| `width` | int | Map width |
| `height` | int | Map height |
| `initial_grid` | int\[\]\[\] \| null | Initial terrain grid for this seed |

</div>

### Error Codes

<div class="table-scroll-wrapper">

| Status | Meaning                                                |
|--------|--------------------------------------------------------|
| 400    | Round not completed/scoring yet, or invalid seed_index |
| 403    | Not on a team                                          |
| 404    | Round not found                                        |

</div>

## GET /astar-island/leaderboard

Public leaderboard. Each team's score is their **best round score of all time** (weighted by round weight).

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>[
  {
    &quot;team_id&quot;: &quot;uuid&quot;,
    &quot;team_name&quot;: &quot;Vikings ML&quot;,
    &quot;team_slug&quot;: &quot;vikings-ml&quot;,
    &quot;weighted_score&quot;: 72.5,
    &quot;rounds_participated&quot;: 3,
    &quot;hot_streak_score&quot;: 78.1,
    &quot;rank&quot;: 1,
    &quot;is_verified&quot;: true
  }
]</code></pre>
</figure>

<div class="table-scroll-wrapper">

| Field | Type | Description |
|----|----|----|
| `weighted_score` | float | Best `round_score × round_weight` across all rounds |
| `rounds_participated` | int | Total rounds this team has submitted predictions |
| `hot_streak_score` | float | Average score of last 3 rounds |
| `is_verified` | bool | Whether all team members are Vipps-verified |
| `rank` | int | Current leaderboard rank |

</div>
