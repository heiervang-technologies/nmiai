# Object Detection - Scoring

## Score Formula

```
score = 0.7 * detection_mAP + 0.3 * classification_mAP
```

Both metrics are computed at **IoU >= 0.5**.

### Detection mAP

- Evaluates the ability to locate objects regardless of category.
- Category labels are **ignored**; only bounding box overlap matters.

### Classification mAP

- Evaluates the ability to correctly classify detected objects.
- Requires both correct bounding box (IoU >= 0.5) **and** correct `category_id`.

## Submission Limits

| Limit | Value |
|-------|-------|
| Submissions per day | 3 |
| Daily reset | Midnight UTC |
| Max concurrent submissions | 2 |
| Infrastructure errors | First 2 do not count against daily limit |

## Final Rankings

Final rankings are determined using a **private test set** that is separate from the public leaderboard test set.

Teams can **manually select** which of their submissions should be used for final evaluation. If no selection is made, the best-scoring public submission is used.
