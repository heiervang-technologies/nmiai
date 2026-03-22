# Remaining CA Transition Rules

## 1. Port Formation
Where do Ports come from?
- From Settle: 2293 (78.3%)
- From Ruin: 635 (21.7%)

For existing Settlements, what is the probability of upgrading to a Port based on Ocean neighbors?
| Ocean Neighbors | P(Upgrade to Port) | Sample Size |
|---|---|---|
| 0 | 0.0000 | 569998 |
| 1 | 0.0043 | 17891 |
| 2 | 0.0638 | 5660 |
| 3 | 0.0727 | 22091 |
| 4 | 0.0706 | 2267 |
| 5 | 0.0579 | 1227 |
| 6 | 0.0622 | 241 |
| 7 | 0.0492 | 61 |

## 2. Forest Clearing
Probability of a Forest turning into a Settlement by Civ Neighbors:
| Civ Neighbors | P(Clear Forest) | Sample Size |
|---|---|---|
| 0 | 0.0019 | 1158809 |
| 1 | 0.0086 | 365071 |
| 2 | 0.0166 | 152365 |
| 3 | 0.0263 | 64577 |
| 4 | 0.0352 | 24466 |
| 5 | 0.0508 | 9386 |
| 6 | 0.0660 | 3394 |
| 7 | 0.0789 | 1014 |
| 8 | 0.1098 | 164 |

## 3. Phase-Dependent Growth Rates
Does the global 4-step cycle change the actual probabilities of growth, or just the number of eligible cells?
Looking at Plains -> Settlement for 1, 2, and 3 civ neighbors across the 4 phases:
| Civ Neighbors | Phase 0 (Growth) | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| 1 | 0.0168 (n=243483) | 0.0056 (n=267358) | 0.0063 (n=264454) | 0.0047 (n=245055) |
| 2 | 0.0330 (n=99523) | 0.0122 (n=106488) | 0.0115 (n=107535) | 0.0094 (n=98589) |
| 3 | 0.0478 (n=40205) | 0.0184 (n=45207) | 0.0179 (n=45854) | 0.0144 (n=40006) |
