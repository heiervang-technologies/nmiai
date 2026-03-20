# Model Comparison: Neighborhood vs Recursive vs Bucket/Predictor Base

## What I checked

I read [benchmark.py](/home/me/ht/nmiai/tasks/astar-island/benchmark.py), which defines the actual offline metric used here:

- `evaluate_predictor()` computes per-cell `KL(p || q)`, multiplies by ground-truth entropy in bits, and averages over dynamic cells only.
- `cross_validate()` does proper leave-one-round-out (LORO): train on all rounds except one, evaluate on the held-out round.

Important caveat: the `benchmark.py` CLI does **not** run CV by default; it only runs in-sample evaluation. So the earlier impressive in-sample numbers for `recursive_model.py` were not evidence of generalization.

## LORO cross-validation results

### 1. `neighborhood_predictor.py`

I ran true LORO CV by rebuilding its lookup tables from the training rounds only.

- CV mean KL: `0.139785`
- CV mean weighted KL: `0.122300`

Per holdout round:

- Round 1: `0.074203`
- Round 2: `0.103067`
- Round 3: `0.132663`
- Round 4: `0.047081`
- Round 5: `0.063740`
- Round 6: `0.264059`
- Round 7: `0.171286`

This model is reasonably robust on easier rounds, but it degrades badly on rounds 6 and 7.

### 2. `recursive_model.py`

I ran LORO CV by training a fresh recursive model for each holdout round and evaluating on that held-out round.

- CV mean KL: `0.125601`
- CV mean weighted KL: `0.100229`

Per holdout round:

- Round 1: `0.060315`
- Round 2: `0.065455`
- Round 3: `0.120644`
- Round 4: `0.058386`
- Round 5: `0.061851`
- Round 6: `0.181923`
- Round 7: `0.153029`

So on this offline CV, the recursive model **does generalize better than the neighborhood predictor**.

## But is the recursive model overfitting?

Yes, still likely.

Why:

- In-sample `recursive_model.json` showed weighted KL around `0.0766`, while LORO CV is `0.1002`. That gap is substantial.
- In [recursive_model.py](/home/me/ht/nmiai/tasks/astar-island/recursive_model.py#L276), when `holdout_round` is set, the held-out round is used as `val_data`.
- In [recursive_model.py](/home/me/ht/nmiai/tasks/astar-island/recursive_model.py#L321), the model repeatedly evaluates on that held-out round and keeps the best checkpoint.

That means the reported LORO CV for `recursive_model.py` is actually **optimistic**, because the holdout round is being used for model selection. Real out-of-sample performance is probably a bit worse than `0.1002`.

So the answer is:

- `recursive_model.py` generalizes **better than** `neighborhood_predictor.py`,
- but it is still overfit relative to its in-sample score,
- and its CV estimate is flattering because of validation leakage.

## Best out-of-sample model right now

I also compared the newer bucket-based predictor family offline using the same benchmark-style `H * KL` metric.

### `predictor.py` base model

- Overall LORO weighted KL: `0.103644`

Per holdout round:

- Round 1: `0.058417`
- Round 2: `0.064909`
- Round 3: `0.138106`
- Round 4: `0.066266`
- Round 5: `0.066636`
- Round 6: `0.176050`
- Round 7: `0.155124`

### Pure bucket priors from the same predictor feature buckets

- Overall LORO weighted KL: `0.104678`

Per holdout round:

- Round 1: `0.059482`
- Round 2: `0.066329`
- Round 3: `0.138193`
- Round 4: `0.066581`
- Round 5: `0.067458`
- Round 6: `0.178011`
- Round 7: `0.156688`

These two are very close, but the conservative `predictor.py` base model edges out pure bucket priors on every holdout round in this offline test.

## Bottom line

1. `recursive_model.py` is **not** the best model to trust operationally just because of its in-sample score. It overfits, and its own CV path is optimistic due to holdout-round checkpoint selection.
2. `neighborhood_predictor.py` generalizes **worse** than the recursive model and is not the best choice.
3. The strongest practical out-of-sample option from what we tested is the **conservative bucket-based predictor family**, specifically `predictor.py` in `base` mode.
4. Pure bucket priors are almost as good, and much safer to reason about. If we want the lowest-risk fallback, pure bucket priors are defensible. But based on offline LORO, `predictor.py` base is slightly better.

## Recommendation

For the next round, I would **not** go back to `recursive_model.py` as the primary submission model.

Best current choice:

- Use `predictor.py --model-config base`

Safe fallback:

- Use pure bucket priors only

I would avoid:

- `recursive_model.py` for primary submissions
- `neighborhood_predictor.py` as the main model

If we keep the recursive approach alive at all, it should only be after fixing the validation leakage and re-running clean CV.
