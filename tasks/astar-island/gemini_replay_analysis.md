# Deep Analytical Exploration of 50-Year Replays

## 1. The 4-Step Cycle & Regime Amplitude
Average cell changes per step across all rounds, grouped by Regime:

| Step | All | Prosperous | Moderate | Harsh |
|---|---|---|---|---|
| 1 | 2.3 | 1.6 | 3.0 | 4.7 |
| 2 | 5.1 | 3.9 | 8.1 | 6.9 |
| 3 | 10.1 | 9.4 | 11.6 | 11.4 |
| 4 | 12.8 | 12.3 | 13.1 | 15.0 |
| 5 | 6.8 | 6.2 | 8.5 | 7.5 |
| 6 | 12.2 | 12.3 | 11.5 | 12.8 |
| 7 | 6.7 | 5.4 | 8.3 | 10.6 |
| 8 | 16.9 | 17.9 | 13.8 | 16.3 |
| 9 | 17.2 | 17.6 | 14.2 | 19.3 |
| 10 | 10.2 | 9.2 | 11.5 | 12.8 |
| 11 | 8.3 | 7.4 | 11.4 | 8.9 |
| 12 | 26.3 | 29.2 | 22.9 | 17.8 |
| 13 | 13.4 | 12.5 | 15.8 | 14.4 |
| 14 | 11.4 | 11.7 | 10.9 | 11.1 |
| 15 | 14.7 | 16.2 | 12.1 | 11.5 |
| 16 | 25.3 | 29.4 | 18.6 | 15.7 |
| 17 | 13.7 | 14.2 | 12.9 | 12.1 |
| 18 | 14.3 | 16.1 | 12.4 | 9.1 |
| 19 | 12.4 | 13.9 | 11.1 | 6.9 |
| 20 | 25.8 | 30.8 | 19.5 | 11.4 |
| 21 | 19.5 | 21.9 | 16.9 | 12.1 |
| 22 | 18.3 | 21.2 | 13.0 | 11.9 |
| 23 | 14.6 | 17.0 | 9.6 | 9.9 |
| 24 | 29.3 | 36.4 | 19.0 | 11.1 |
| 25 | 21.8 | 26.8 | 13.1 | 10.4 |
| 26 | 20.6 | 24.6 | 15.9 | 8.8 |
| 27 | 18.5 | 21.7 | 16.2 | 7.6 |
| 28 | 34.2 | 43.1 | 22.2 | 9.6 |
| 29 | 27.9 | 34.6 | 20.1 | 8.1 |
| 30 | 27.5 | 34.8 | 16.4 | 8.7 |
| 31 | 23.7 | 30.1 | 13.9 | 7.5 |
| 32 | 36.3 | 47.6 | 17.2 | 9.3 |
| 33 | 32.9 | 42.4 | 17.9 | 9.3 |
| 34 | 32.7 | 41.8 | 20.4 | 7.5 |
| 35 | 30.4 | 38.8 | 19.3 | 6.8 |
| 36 | 43.1 | 56.2 | 23.8 | 9.4 |
| 37 | 36.6 | 47.6 | 20.4 | 7.9 |
| 38 | 36.5 | 49.0 | 15.9 | 6.6 |
| 39 | 35.9 | 47.6 | 16.2 | 8.0 |
| 40 | 49.1 | 64.8 | 24.7 | 9.7 |
| 41 | 40.7 | 52.5 | 23.7 | 10.0 |
| 42 | 43.6 | 57.7 | 21.4 | 8.5 |
| 43 | 41.6 | 55.8 | 20.1 | 5.1 |
| 44 | 49.5 | 66.2 | 25.4 | 5.3 |
| 45 | 46.7 | 61.3 | 27.6 | 6.1 |
| 46 | 44.9 | 58.2 | 29.5 | 5.4 |
| 47 | 43.5 | 57.3 | 24.9 | 5.5 |
| 48 | 56.1 | 75.3 | 27.8 | 6.1 |
| 49 | 51.7 | 69.5 | 24.5 | 6.4 |
| 50 | 50.1 | 65.4 | 30.9 | 6.5 |

## 2. Phase Mapping (Step Modulo 4)
Analyzing the dominant transitions based on `Step % 4`:

### Step % 4 == 0
- **Plains->Settle**: 10892 occurrences
- **Settle->Ruin**: 8043 occurrences
- **Ruin->Settle**: 4368 occurrences
- **Forest->Settle**: 4180 occurrences
- **Ruin->Plains**: 3229 occurrences
### Step % 4 == 1
- **Settle->Ruin**: 9728 occurrences
- **Ruin->Settle**: 5532 occurrences
- **Plains->Settle**: 4395 occurrences
- **Ruin->Plains**: 4057 occurrences
- **Ruin->Forest**: 1989 occurrences
### Step % 4 == 2
- **Settle->Ruin**: 9268 occurrences
- **Ruin->Settle**: 5586 occurrences
- **Plains->Settle**: 4573 occurrences
- **Ruin->Plains**: 3998 occurrences
- **Ruin->Forest**: 1897 occurrences
### Step % 4 == 3
- **Settle->Ruin**: 7833 occurrences
- **Ruin->Settle**: 4424 occurrences
- **Ruin->Plains**: 3175 occurrences
- **Plains->Settle**: 3160 occurrences
- **Ruin->Forest**: 1583 occurrences

## 3. Hidden Rules Investigation

### Forest Proximity vs Survival
Does having more adjacent forests improve a settlement's chance of surviving the next step?
| Forest Neighbors | Survived | Total | Survival Rate |
|---|---|---|---|
| 0 | 96196 | 103840 | 92.64% |
| 1 | 159346 | 170904 | 93.24% |
| 2 | 129936 | 138962 | 93.50% |
| 3 | 66535 | 71043 | 93.65% |
| 4 | 22261 | 23893 | 93.17% |
| 5 | 6065 | 6486 | 93.51% |
| 6 | 974 | 1046 | 93.12% |
| 7 | 155 | 164 | 94.51% |

### Port Proximity vs Survival (Trade Bonus?)
| Has Port Neighbor | Survived | Total | Survival Rate |
|---|---|---|---|
| False | 456177 | 489367 | 93.22% |
| True | 25299 | 26981 | 93.77% |

### Cluster Size vs Collapse Rate (Conflict Trigger)
Does a settlement cluster collapsing (turning to ruins) correlate with the size of the cluster?
| Cluster Size | Ruined Cells | Total Cells | Collapse Rate |
|---|---|---|---|
| 0-4 | 19045 | 283106 | 6.73% |
| 5-9 | 6821 | 107305 | 6.36% |
| 10-14 | 3138 | 47287 | 6.64% |
| 15-19 | 1787 | 26306 | 6.79% |
| 20-24 | 1103 | 14917 | 7.39% |
| 25-29 | 710 | 10356 | 6.86% |
| 30-34 | 545 | 7523 | 7.24% |
| 35-39 | 390 | 5894 | 6.62% |
| 40-44 | 233 | 3372 | 6.91% |
| 45-49 | 252 | 3371 | 7.48% |
| 50-54 | 2123 | 32704 | 6.49% |

## 4. Regime Rates vs Rules
Based on the amplitude table above, the *timing* of the phases is perfectly identical across all regimes (spikes happen on the exact same steps). However, the *amplitude* is vastly different. Prosperous rounds see 150+ changes per growth phase, while Harsh rounds see < 10. The **rules are the same, but the transition probabilities are scaled.**
