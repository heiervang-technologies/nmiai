# Astar Island Automaton Analysis

This note reverse-engineers likely simulator rules from the five `round1_seed*.json` ground-truth files. The evidence comes from the full 40x40 initial grids, the 40x40x6 final probability tensors, and the visualizations generated in `tasks/astar-island/visualizations/`.

## What looks deterministic

- `Ocean -> Empty` is effectively fixed. Across 1,016 ocean cells over the five seeds, the final distribution is `P(Empty)=1.0000`.
- `Mountain -> Mountain` is fixed. Across 171 mountain cells, the final distribution is `P(Mountain)=1.0000`.
- This strongly suggests the dynamic automaton only updates traversable land cells, while ocean and mountain are static terrain constraints.

## Core land transitions

Average transition probabilities across all five seeds:

- `Plains -> Empty 0.7845, Settlement 0.1557, Port 0.0140, Ruin 0.0121, Forest 0.0336`
- `Forest -> Forest 0.7449, Settlement 0.1588, Port 0.0142, Ruin 0.0124, Empty 0.0697`
- `Settlement -> Settlement 0.4100, Empty 0.3705, Forest 0.1805, Ruin 0.0311, Port 0.0078`
- `Port -> Port 0.3186, Empty 0.3643, Forest 0.1757, Settlement 0.1200, Ruin 0.0214`

Likely implications:

- Plains and forest are the main growth substrate. Both can become settlements at similar rates, but forest has a much stronger tendency to remain forest.
- Existing settlements are not stable enough to simply persist forever. They frequently collapse to empty land, and a smaller fraction decays into forest or ruin.
- Ports are rare, coastal, and fragile. Even when a cell starts as a port, persistence is only about `0.32` on average.
- Ruins are a low-probability state in final distributions, which suggests raids or collapse occur, but the environment phase later cleans many ruined cells back into empty land or forest.

## Settlement growth rule hypotheses

The strongest regularity is that settlement probability rises near existing settlements.

- For all plains cells: `P(Settlement)=0.1557`
- For plains adjacent to an initial settlement: `P(Settlement)=0.2263`
- Per seed, plains with 1 adjacent initial settlement land around `0.206-0.246` settlement probability.
- Plains with 2 adjacent initial settlements are even higher, around `0.214-0.318` depending on seed.
- Forest next to an initial settlement also rises to `P(Settlement)=0.2295`, versus `0.1588` for forest overall.

Likely rule:

- During the Growth phase, empty/plains/forest cells receive a settlement-growth bonus from nearby settlements.
- The bonus appears local and monotone with neighborhood support. An 8-neighbor interaction fits the observed pattern well.
- Forest seems colonizable, not blocked terrain. It probably either gets cleared directly into settlement or first into empty and then populated within the same yearly cycle.

## Port formation rule hypotheses

Ports are spatially the cleanest inferred rule.

- Coastal plains have `P(Port)=0.0921`, versus `0.0140` for plains overall.
- Non-coastal land has essentially zero high-probability port cells in the inspected seeds.
- Every cell with `P(Port) > 0.1` lies on the coast in all five seeds.
- Those high-port cells are mostly coastal plains, with a smaller number of coastal forest cells and a few coastal settlement/port cells.
- Initial coastal settlements are especially port-prone: `P(Port)=0.3938` on average, though there are only 4 such cells in these seeds.

Likely rule:

- During the Trade phase, ports can only form on land cells adjacent to ocean.
- A nearby settlement is probably required or at least strongly helpful.
- Ports appear to be an economic specialization of coastal settlements rather than an independent terrain class.

## Conflict and ruin rule hypotheses

Ruin exists, but only as a weak final-state mode.

- Overall ruin mass is about `1-1.3%` depending on seed.
- Initial settlements have elevated ruin probability: `P(Ruin)=0.0311`.
- Plains near initial settlements rise to `P(Ruin)=0.0158`, versus `0.0121` overall.
- Forest near settlements also rises slightly to `P(Ruin)=0.0166`.
- No cell in these five seeds reaches ruin probability above `0.1`; max ruin probability stays around `0.055-0.100`.

Likely rule:

- Conflict is real but not dominant. Raids or warfare probably target settlement clusters and their frontier.
- Ruins are transient. The Environment phase likely converts many ruins into empty land or forest quickly, preventing ruin from dominating argmax outcomes.
- The fact that ruin probability is highest around settlements but rarely wins argmax suggests many simulations create ruins there, but just as many later reclaim them.

## Winter / collapse hypotheses

Settlement collapse is also visible.

- Initial settlements end as `Empty` with probability `0.3705`.
- Initial ports end as `Empty` with probability `0.3643`.
- This is too large to be explained by conflict alone, especially given low final ruin mass.

Likely rule:

- Winter probably applies a survival penalty to settlements and ports unless they satisfy local support conditions such as food, trade access, or enough neighboring civilization.
- Cells that fail winter do not necessarily become ruins; many appear to revert directly to empty land.

## Environment / reclamation hypotheses

The Environment phase is visible through rewilding.

- Initial settlements become forest with probability `0.1805`.
- Initial ports become forest with probability `0.1757`.
- Forest is sticky overall: `P(Forest)=0.7449` from initial forest.

Likely rule:

- Abandoned or ruined civilization tiles are reclaimed by forest at a meaningful rate.
- This reclamation is probably stronger on previously forested cells and weaker on open plains.
- The simulator likely alternates human expansion and environmental recovery, rather than allowing only one-way civilization growth.

## Spatial pattern summary

- Settlement growth is clustered. High settlement probability fields form halos around initial settlement groups rather than isolated random dots.
- Port probability traces coastlines and fjords, not inland corridors.
- Ruin probability is diffuse and settlement-adjacent, not concentrated in remote areas.
- Entropy is highest near active civilization fronts. Mean entropy near initial settlements is about `1.09-1.18` bits, versus `0.63-0.84` far away.
- Seed-to-seed variation mostly changes how strongly a given frontier expands or collapses, not the overall geography of where growth is allowed.

## Best current automaton sketch

The simulator is plausibly doing something close to this each year:

1. Growth: settlements expand into adjacent plains and some forest, with probability increasing with nearby settlements.
2. Conflict: settlement-heavy zones sometimes collapse into ruin or empty land.
3. Trade: coastal settlement-adjacent land can convert into ports.
4. Winter: weak or isolated settlements/ports die back, often directly to empty.
5. Environment: ruins and abandoned land are partially reclaimed into forest.

## Practical modeling takeaways

- Treat ocean and mountain as deterministic.
- Use distance to initial settlements as a major feature for settlement probability.
- Gate port probability on coastal adjacency first; inland port mass should be near zero.
- Keep ruin probability nonzero near settlement clusters, but small.
- Include a reclamation pathway from settlement/port back to forest, not just to empty.
- Expect broad uncertainty bands around settlement frontiers rather than around static terrain.

## Files produced

- `tasks/astar-island/visualizations/seed{0..4}_initial.png`
- `tasks/astar-island/visualizations/seed{0..4}_argmax.png`
- `tasks/astar-island/visualizations/seed{0..4}_entropy.png`
- `tasks/astar-island/visualizations/seed{0..4}_overview.png`
- `tasks/astar-island/visualizations/seed{0..4}_transitions.png`
- `tasks/astar-island/visualizations/seed{0..4}_patterns.png`
- `tasks/astar-island/visualizations/round1_cross_seed_means.png`
- `tasks/astar-island/visualizations/round1_seed_metrics.png`
- `tasks/astar-island/visualizations/round1_summary_stats.txt`
