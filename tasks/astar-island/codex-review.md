- [solver.py](/home/me/ht/nmiai/tasks/astar-island/solver.py#L167) `settlement_dist` only uses initial settlements (`init == 1`) and ignores initial ports (`init == 2`). That makes cells around ports too cold even though ports are also dynamic civilization centers. Build the distance transform from `(init == 1) | (init == 2)`.
- [solver.py](/home/me/ht/nmiai/tasks/astar-island/solver.py#L175) `forest_interior` is not detecting interior forest correctly. On the current round it marks essentially no forest cells as interior, so almost every forest cell falls back to the edge prior/tau. The current expression also treats ocean/mountain adjacency oddly. Replace it with a direct erosion of `forest_mask` using the neighborhood you actually want (likely 4-neighbor cross; 8-neighbor only if you intentionally want a stricter definition).

Minor notes:

- [solver.py](/home/me/ht/nmiai/tasks/astar-island/solver.py#L264) The posterior calculation itself is correct: `posterior = counts + tau * prior_mean`.
- [solver.py](/home/me/ht/nmiai/tasks/astar-island/solver.py#L232) The `tau` values are broadly sensible once the hot-zone distance and forest-interior mask are fixed.
