# Astar Island - API Endpoints

## Base URL

```
https://api.ainm.no/astar-island
```

## Authentication

- **JWT Cookie**: Set via the competition platform login
- **Bearer Token**: Pass as `Authorization: Bearer <token>` header

## Endpoints

### GET /rounds

List all available rounds.

### GET /rounds/{id}

Get details for a specific round including seed information and current year.

### GET /leaderboard

View the current leaderboard.

### GET /budget

Check remaining query budget for the current round.

### POST /simulate

Query the simulator to observe a viewport of the island.

**Request:**

```json
{
  "round_id": 1,
  "seed_index": 0,
  "viewport_x": 10,
  "viewport_y": 10,
  "viewport_w": 15,
  "viewport_h": 15
}
```

### POST /submit

Submit predictions for a seed.

**Request:**

```json
{
  "round_id": 1,
  "seed_index": 0,
  "prediction": [
    [[0.1, 0.2, 0.3, 0.1, 0.2, 0.1], ...],
    ...
  ]
}
```

The prediction tensor has shape `[40][40][6]`.

### GET /my-rounds

List rounds you have participated in.

### GET /my-predictions/{round_id}

View your submitted predictions for a round.

### GET /analysis/{round_id}/{seed_index}

Get analysis of your prediction for a specific seed (available after round closes).
