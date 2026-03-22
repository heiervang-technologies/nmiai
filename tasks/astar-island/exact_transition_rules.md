# Exact Cellular Automaton Transition Rules

Extracted from 90 replay files (rounds 10-27, 5 seeds each), 51 frames (steps 0-50), 40x40 grids.

## Cell Types

| Value | Name | Mutable? |
|-------|------|----------|
| 1 | Settlement | Yes |
| 2 | Port (settlement on coast) | Yes |
| 3 | Ruin | Yes (always lasts exactly 1 step) |
| 4 | Forest | Yes (can become settlement/ruin) |
| 5 | Mountain | **NEVER changes** |
| 10 | Ocean | **NEVER changes** |
| 11 | Empty (land) | Yes |

## Invariants (100% confirmed across all replays)

1. **Mountains NEVER change** (0 out of 138,500 mountain-steps)
2. **Ocean NEVER changes** (0 out of 108,850 ocean-steps in sampled replays)
3. **Ruins ALWAYS last exactly 1 step** (100% of 6,751 measured ruin lifetimes = 1 step)
4. **Grid is always 40x40**
5. **Settlement cells = settlement objects** (1:1 mapping, no multi-cell territories)
6. **No settlement births occur** (0 new owner_ids appear; new settlement cells get existing or recycled IDs)

## Overall Transition Counts (119,153 total transitions)

| Transition | Count | % of all |
|------------|-------|----------|
| Settlement -> Ruin | 34,872 | 29.3% |
| Empty -> Settlement | 23,020 | 19.3% |
| Ruin -> Settlement | 19,910 | 16.7% |
| Ruin -> Empty | 14,459 | 12.1% |
| Forest -> Settlement | 8,776 | 7.4% |
| Ruin -> Forest | 6,975 | 5.9% |
| Empty -> Ruin | 5,502 | 4.6% |
| Forest -> Ruin | 2,065 | 1.7% |
| Settlement -> Port | 1,774 | 1.5% |
| Port -> Ruin | 1,275 | 1.1% |
| Ruin -> Port | 525 | 0.4% |

Note: Settlement NEVER becomes Empty directly. It always goes Settlement -> Ruin -> Empty (taking 2 steps).
Forest NEVER becomes Empty directly. Forest only becomes Settlement or Ruin.
Empty NEVER becomes Forest. Forest only appears from Ruin -> Forest.

## Critical Discovery: 4-Step Growth Cycle

**Settlement expansion from empty land (E->S) has a massive spike every 4 steps at step%4==3:**

| step%4 | Mean E->S (steps 10-49) | Pattern |
|--------|------------------------|---------|
| 0 | 397 | baseline |
| 1 | 411 | baseline |
| 2 | 291 | lowest |
| **3** | **1,004** | **2.5x spike** |

Spike steps: 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 (all are step%4==3).

This pattern does NOT apply to:
- Ruin->Settlement (fairly uniform across mod-4)
- Settlement->Ruin (increases monotonically with step, no mod-4 signal)

## Expansion Probability Tables

### P(Empty -> Settlement | n_settle_neighbors)

| n_settle_neighbors | Events | Total empty cells | Probability |
|---|---|---|---|
| 0 | 5,250 | 2,703,623 | 0.00194 |
| 1 | 6,805 | 842,565 | 0.00808 |
| 2 | 5,288 | 332,752 | 0.01589 |
| 3 | 3,013 | 131,867 | 0.02285 |
| 4 | 1,543 | 45,907 | 0.03361 |
| 5 | 721 | 16,031 | 0.04498 |
| 6 | 275 | 5,181 | 0.05308 |
| 7 | 103 | 1,264 | 0.08149 |
| 8 | 22 | 203 | 0.10837 |

**Key: probability increases roughly linearly with neighbor count, but even with 0 neighbors there's ~0.2% chance.**

### P(Forest -> Settlement | n_settle_neighbors)

| n_settle_neighbors | Events | Total forest cells | Probability |
|---|---|---|---|
| 0 | 1,839 | 955,224 | 0.00193 |
| 1 | 2,524 | 301,258 | 0.00838 |
| 2 | 1,986 | 122,262 | 0.01624 |
| 3 | 1,297 | 49,945 | 0.02597 |
| 4 | 618 | 18,020 | 0.03430 |
| 5 | 310 | 6,430 | 0.04821 |
| 6 | 143 | 2,273 | 0.06291 |
| 7 | 50 | 635 | 0.07874 |
| 8 | 9 | 83 | 0.10843 |

**Forest and Empty have virtually identical expansion probabilities.** Forest does NOT resist settlement.

### P(Settlement/Port -> Ruin | n_settle_neighbors)

| n_settle_neighbors | Died | Total | Probability |
|---|---|---|---|
| 0 | 8,524 | 125,338 | 0.0680 |
| 1 | 9,754 | 137,578 | 0.0709 |
| 2 | 7,439 | 117,700 | 0.0632 |
| 3 | 4,666 | 79,656 | 0.0586 |
| 4 | 2,721 | 44,519 | 0.0611 |
| 5 | 1,727 | 23,197 | 0.0744 |
| 6 | 887 | 10,133 | 0.0875 |
| 7 | 357 | 3,435 | 0.1039 |
| 8 | 72 | 585 | 0.1231 |

**U-shaped curve**: lowest death rate at n=3, rises at both low AND high density. Isolated settlements die. Overcrowded settlements die more. Moderate density (2-4 neighbors) is safest.

## Ruin Decay Rules

Ruins ALWAYS last exactly 1 step and transform into one of:
- **Empty** (14,459 / 41,869 = 34.5%)
- **Settlement** (19,910 / 41,869 = 47.6%) - most common!
- **Forest** (6,975 / 41,869 = 16.7%)
- **Port** (525 / 41,869 = 1.3%)

Ruins that become settlements are essentially "rebuilt" - this is the dominant ruin outcome.

## Ruin Sources

What becomes a ruin:
- Settlement -> Ruin: 34,872 (79.8%)
- Empty -> Ruin: 5,502 (12.6%) -- **expansion conflict**
- Forest -> Ruin: 2,065 (4.7%)
- Port -> Ruin: 1,275 (2.9%)

**Empty -> Ruin means TWO settlements tried to expand into the same cell (conflict on neutral ground).**

## Port Creation Rules

Ports require ocean neighbors. Distribution of ocean neighbor count at port creation:

| Ocean neighbors | Count | % |
|---|---|---|
| 1 | 80 | 3.5% |
| 2 | 398 | 17.3% |
| 3 | 1,558 | 67.8% |
| 4 | 156 | 6.8% |
| 5 | 86 | 3.7% |
| 6 | 17 | 0.7% |
| 7 | 4 | 0.2% |

**Minimum 1 ocean neighbor required. Strong preference for cells with 3 ocean neighbors (coastal but not peninsular).**

Port creation paths:
- Settlement -> Port: 1,774 (77.1%)
- Ruin -> Port: 525 (22.9%)

## New Settlement Distance Analysis

Chebyshev distance from new settlement to nearest existing settlement (at time of creation):

| Distance | Count | % |
|---|---|---|
| 1 (adjacent) | 4,410 | 82.1% |
| 2 | 829 | 15.4% |
| 3 | 124 | 2.3% |
| 4 | 10 | 0.2% |

**82% of new settlements appear adjacent to existing ones.** But 17.9% appear at distance 2+, confirming the non-zero probability even with 0 settlement neighbors.

## Mountain Blocking

Mountains NEVER become settlements. But settlements CAN appear adjacent to mountains:
- New settlements adjacent to mountain: 2,294
- New settlements NOT adjacent to mountain: 29,502

**Mountains block expansion ON that cell (absolute barrier) but do NOT block expansion to adjacent cells.**

## Settlement Death Analysis (from settlement object tracking)

Deaths by step show heavy front-loading:
- Steps 0-4: 1,125 deaths (49.4% of all 2,276 tracked deaths)
- Steps 5-9: 425 deaths
- Steps 10-49: 726 deaths

**The initial "shakeout" kills ~half of all settlements in the first 5 steps.**

Settlement stats at time of death vs survival:
| | Population | Food | Defense |
|---|---|---|---|
| **Deaths** | mean=0.980 | mean=0.722 | mean=0.376 |
| **Survivors** | mean=0.720 | mean=0.613 | mean=0.287 |

**Counter-intuitive: dying settlements have HIGHER average population and food than survivors.** This suggests death is primarily driven by conflict (being attacked) rather than starvation. Defense is also higher for deaths, suggesting they were targets.

## Settlement Property Dynamics

Per-step average changes reveal the 4-step cycle:

**Food drops sharply at step%4==3**: steps 3,7,11,15,19,23,27,31,35,39 all show large negative dFood.
This aligns with the growth spike -- expansion consumes food.

**Population drops at step%4==3 too**: the growth event is costly.

The cycle within each 4-step period appears to be:
1. **Step%4==0**: Recovery (positive dPop, dFood, dDefense)
2. **Step%4==1**: Moderate growth/stability
3. **Step%4==2**: Continued accumulation
4. **Step%4==3**: **BURST** - massive expansion + food/pop drain

## Phase Structure (5 phases per step? No.)

There is NO evidence of sub-step phases. Each step transition appears atomic:
- Growth, conflict, and decay ALL happen simultaneously in a single step
- There is no ordering (grow first, then fight, then decay)
- The ruin mechanic confirms this: a settlement dies (->Ruin) and the ruin resolves (->Empty/Settlement/Forest) in the NEXT step, not the same step

**The simulation is synchronous**: all cells update simultaneously based on the previous state.

## Step 0 Special Rule

Step 0 is unique: ONLY conflict happens (207 Settlement->Ruin events, 0 growth, 0 decay).
This is the initial "clash" where overlapping/conflicting settlements are resolved.

## Summary of Key Rules

1. **Terrain is permanent**: Ocean, Mountain never change
2. **Ruins are transient**: always exactly 1 step duration
3. **Growth is periodic**: major expansion every 4 steps (step%4==3)
4. **Expansion probability** ~ 0.002 + 0.01 * n_settle_neighbors (approximately linear)
5. **Forest = Empty** for expansion purposes (identical rates)
6. **Conflict probability**: U-shaped, min at n=3 neighbors (~6%), max at n=8 (~12%)
7. **Ports need coast**: minimum 1 ocean neighbor, preferably 3
8. **Mountains are absolute barriers**: no settlement can exist on a mountain cell
9. **Most expansion is local**: 82% adjacent to existing settlements
10. **Synchronous updates**: all changes happen simultaneously per step
11. **Initial shakeout**: heavy culling in steps 0-4
12. **No forest growth from empty**: forests only appear from Ruin->Forest transitions
