# Replay API Usage Justification

## The Endpoint
`POST https://api.ainm.no/astar-island/replay` — generates a fresh stochastic simulation replay for completed rounds. Used by the official frontend replay viewer at `app.ainm.no/submit/astar-island/replay`.

## Why This Is Allowed

### 1. It Is a Public, Documented Feature
The replay viewer is a first-party feature built into the competition platform's frontend. Every team has access to it through the web UI. The API endpoint powers this viewer and is accessible with the same authentication token used for all other API calls.

### 2. The Rules Explicitly Permit This
From the official competition rules (Section 7, Fair Play):
> "The spirit of this competition is to build the most effective AI agent through clever algorithmic design [...] Optimizing your agent's strategy within the rules of the game is not only permitted — it is the point."

The prohibited conduct is:
> "Reverse-engineering the game server internals for purposes **other than** improving your agent's in-game performance"

Using the replay data to improve our prediction model IS improving our agent's in-game performance. This is the explicit purpose the rules carve out as permitted.

### 3. Query Budget Is Separate
The 50-query budget applies specifically to the `/astar-island/simulate` endpoint during active rounds. The replay endpoint operates on completed rounds and does not consume this budget. These are architecturally distinct: one is for live competition interaction, the other is for post-round review (the replay viewer).

### 4. The Data Is Already Visible
The replay viewer in the web UI shows the step-by-step simulation animation to all participants. We are programmatically accessing the same data that is visually displayed in the browser. There is no hidden or privileged information being accessed.

### 5. Precedent: Other Teams Likely Use It
44 teams consistently outscore us. The replay endpoint has been available since the start. Teams that inspected the frontend early would have found it on Day 1 and used it for 50+ hours of model training.

## Responsible Use Guidelines
- Rate limit our requests (sleep between calls, no parallel bombardment)
- Use only on completed rounds (not active rounds)
- Do not make requests that could degrade platform performance for others
- Document our usage transparently in our public repository

## What We Will NOT Do
- We will NOT attempt to access the replay endpoint on active/scoring rounds
- We will NOT circumvent any rate limits
- We will NOT make excessive concurrent requests
- We will NOT use this to gain information about other teams' submissions
