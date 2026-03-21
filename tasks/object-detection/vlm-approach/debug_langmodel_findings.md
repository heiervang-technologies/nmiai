# Language Model Debug Findings (Updated)

## Executive Conclusion

The pure PyTorch language model implementation is **correct**.

- `torch_chunk_gated_delta_rule` in `submission-markusnet/run_fast.py` matches transformers exactly in direct function tests (`max_abs=0.0`, `mean_abs=0.0`, cosine ~1.0).
- Layer-by-layer LM parity vs transformers is effectively exact through the decoder stack when using identical checkpoint weights and identical multimodal `inputs_embeds` + `position_ids`.
- The suspected Gated DeltaNet bug is **not** the root cause of the large accuracy gap.

## What Was Actually Fixed

### 1) Qwen preprocessing alignment (critical)
Updated submission implementation to match Qwen3.5 processor behavior:

- `QWEN_MEAN = [0.5, 0.5, 0.5]`
- `QWEN_STD = [0.5, 0.5, 0.5]`
- `QWEN_MIN_PIXELS = 65536`
- `QWEN_MAX_PIXELS = 16777216`
- `smart_resize()` behavior aligned with HF

Files:
- `tasks/object-detection/submission-markusnet/run_fast.py`
- `tasks/object-detection/submission-markusnet/run.py`

### 2) Removed non-reference DeltaNet behavior
- Removed/avoided forcing `chunk_size=seq_len` override in linear attention path; default chunk behavior retained.

File:
- `tasks/object-detection/submission-markusnet/run_fast.py`

### 3) Vision dtype safety in submission `run.py`
Added explicit cast in vision encoder forward path:

- `pixel_values = pixel_values.to(self.patch_embed_proj.weight.dtype)`

File:
- `tasks/object-detection/submission-markusnet/run.py`

## Hard Evidence Collected

### A) DeltaNet kernel parity (direct function-level)
HF vs our `torch_chunk_gated_delta_rule`:
- `max_abs = 0.0`
- `mean_abs = 0.0`
- cosine ~ `1.0`

Chunk-size impact (64 vs seq_len) on same function is tiny:
- `max_abs ~ 2e-7`
- not remotely large enough to explain a 44% absolute accuracy collapse.

### B) Layer parity script
Added:
- `tasks/object-detection/vlm-approach/debug_lm_layer_parity.py`

Observed output on real image input:
- Decoder layer outputs (`layer_out_00` to `layer_out_11`) match with cosine ~1.0 and `max_abs=0.0`.
- Final normalized hidden state matches (`final_cos ~1.0`, `final_max_abs=0.0`).

Note:
- A low cosine at hidden-state index `12` appears as an indexing/reporting artifact against HF hidden state slots, not an actual final LM mismatch (confirmed by exact final-state match).

## Operational Observations

- Full local end-to-end eval in this environment is very slow for pure-PyTorch MarkusNet on CPU fallback because each image produces many detections, and each crop runs full VLM classification.
- This slowdown affects iteration speed but does not change the core finding: LM math is correct.

## Recommended Team Direction

1. Stop allocating debugging time to DeltaNet correctness.
2. Focus on:
- preprocessing/data path parity,
- runtime constraints (crop count / classification throughput),
- stronger eval signal for candidate model variants.
3. Keep `debug_lm_layer_parity.py` as a regression check whenever modifying LM or preprocessing paths.
