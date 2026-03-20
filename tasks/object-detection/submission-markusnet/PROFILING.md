# MarkusNet Inference Pipeline Profiling

## Test Setup
- **GPU**: NVIDIA GeForce RTX 3090 (24GB) — submission target is L4 (24GB)
- **Test images**: 3 images from `/tmp/test_3imgs/`
- **Detections**: 190 + 136 + 88 = 414 crops total (avg 138/image)
- **YOLO ran on CPU** (onnxruntime CUDA libs missing in venv) — on L4 this will be faster

## Raw Timing Results (3 images, 414 crops)

| Stage | Total (s) | Count | Avg (s) |
|-------|-----------|-------|---------|
| 1. YOLO load | 0.411 | 1 | 0.411 |
| 2. MarkusNet init | 1.837 | 1 | 1.837 |
| 3. NF4 dequant + load | 1.115 | 1 | 1.115 |
| 4. Model to device | 0.309 | 1 | 0.309 |
| 5. YOLO preprocess | 0.036 | 3 | 0.012 |
| 6. YOLO inference | 7.395 | 3 | 2.465 |
| 7. YOLO postprocess | 0.034 | 3 | 0.012 |
| 8. Crop extraction | 0.043 | 3 | 0.014 |
| 9. Vision preprocess | 2.128 | 414 | 0.0051 |
| 10. **Vision encoder** | **12.616** | 414 | **0.0305** |
| 11. Embed scatter | 2.273 | 414 | 0.0055 |
| 12. **Language model** | **31.966** | 414 | **0.0772** |
| 13. Cls head | 1.184 | 414 | 0.0029 |
| **TOTAL** | **61.348** | | |

## Bottleneck Analysis

### The problem: crops processed ONE AT A TIME

The `classify_crops()` method iterates `for crop in batch_crops` (line 918) and runs **the full vision encoder + language model independently per crop**. There is NO batching at the model level despite the outer batch loop.

Each crop goes through:
1. Vision preprocess: resize to 448x448, create patches (5.1ms)
2. **Vision encoder**: 12 ViT blocks on 784 tokens (30.5ms)
3. Embed scatter + position IDs (5.5ms)
4. **Language model**: 12 hybrid blocks (8 linear attn + 3 full attn) on ~792 tokens (77.2ms)
5. Classification head (2.9ms)

**Total per crop: 121.2ms**

### Extrapolation to 200 images (~27,600 crops at 138/image)

| Component | Estimated time |
|-----------|---------------|
| Model loading (one-time) | 3.7s |
| YOLO (200 images, GPU) | ~10-20s |
| MarkusNet per-crop (27,600 crops) | **3,344s** |
| **Total** | **~3,360s** |

**The pipeline is ~13x too slow.**

### Where time is spent (per crop)

- **Language model: 63.7%** — 77.2ms/crop. The chunked gated delta rule (linear attention) is the main cost. It runs a Python for-loop over chunks (line 753: `for i in range(0, total_sequence_length // chunk_size)`).
- **Vision encoder: 25.1%** — 30.5ms/crop. 12 ViT blocks with full attention on 784 tokens.
- **Vision preprocess: 4.2%** — PIL resize + numpy normalization.
- **Embed scatter: 4.5%** — Token embedding + masked_scatter.
- **Cls head: 2.4%** — Trivial linear layers.

## Proposed Optimizations

### Tier 1: Essential (must-do to get under 300s)

#### 1. Reduce crop resolution: 448 -> 224 (estimated 4x speedup on vision, ~2x on LM)
- Currently: 448x448 -> 28x28 patches -> 784 tokens -> 196 merged tokens
- At 224x224: 14x14 patches -> 196 tokens -> 49 merged tokens
- Vision encoder attention is O(n^2), so 784->196 = ~16x fewer FLOPs in attention
- Language model sequence drops from ~204 to ~57 tokens
- **Risk**: May hurt classification accuracy. Test first.
- **Estimated per-crop**: ~30ms (from 121ms) -> 200 images in ~830s still too slow alone

#### 2. True batched inference (estimated 5-10x throughput increase)
- Current code processes each crop independently through vision + LM
- Batch vision encoder: stack all 784-token sequences, run all 12 ViT blocks at once
- Batch language model: pad sequences, run with attention mask
- On GPU, batch=64 should be ~5-10x faster than batch=1 due to better utilization
- **Estimated per-crop**: ~12ms amortized -> 200 images in ~335s (still tight)

#### 3. Combine resolution reduction + batching (target: under 250s)
- 224x224 crops + batch_size=64 should give ~3-5ms amortized per crop
- 200 images * 138 crops = 27,600 crops * 3ms = 83s + 20s YOLO + 4s loading = **~107s**

### Tier 2: High-impact optimizations

#### 4. torch.compile (estimated 1.5-2x speedup)
- `model = torch.compile(model, mode="reduce-overhead")`
- Fuses operations, reduces kernel launches
- Works well with the custom attention code
- **Risk**: Compilation takes 30-60s on first run, which cuts into the 300s budget
- Consider: compile with `mode="max-autotune"` offline, save compiled model

#### 5. Replace linear attention Python loop with fused kernel
- `torch_chunk_gated_delta_rule` has a Python for-loop (line 753) iterating over chunks
- This prevents GPU parallelism and adds Python overhead
- Options:
  - Use `fla` (Flash Linear Attention) library's CUDA kernel if available
  - Rewrite loop with torch.vmap or compile-friendly form
- **Estimated**: 2-3x speedup on language model alone

#### 6. FP16 instead of BF16
- L4 has better FP16 throughput than BF16 (unlike A100/H100)
- Change `dtype = torch.float16`
- **Estimated**: 10-20% speedup on L4

### Tier 3: Larger changes

#### 7. Export to ONNX + CUDAExecutionProvider
- Export entire MarkusNet to ONNX, run with onnxruntime GPU
- Eliminates Python overhead entirely
- **Challenge**: Custom linear attention (delta rule) may not export cleanly
- **Estimated**: 2-3x speedup if it works

#### 8. Prune to fewer transformer blocks
- Currently: 12 vision + 12 language = 24 blocks
- Prune to 8 vision + 8 language = 16 blocks
- **Requires retraining** the classification head on pruned model
- **Estimated**: ~33% reduction in compute

#### 9. Skip language model entirely
- Use only vision encoder output + classification head
- Retrain cls_head on vision embeddings directly (196 tokens * 1024 dim)
- Eliminates 63.7% of per-crop time
- **Requires retraining**

### Tier 4: Quick wins

#### 10. YOLO: Ensure GPU execution
- Current profiling had YOLO on CPU (2.5s/image)
- With CUDA provider: ~50ms/image expected
- Saves ~490s on 200 images

#### 11. Pre-allocate tensors
- `_build_position_ids` creates new tensors every call
- Pre-compute for the fixed 448x448 (or 224x224) case, reuse across crops

#### 12. Reduce MAX_DET from 300 to 100
- Currently 138 avg detections, max 300
- Most images likely have <100 real products
- Fewer crops = proportionally less MarkusNet time

## Recommended Action Plan

**Priority order to hit 250s target on L4:**

1. **Reduce crop size to 224x224** (code change: 1 line)
2. **Implement true batched vision + language forward** (medium effort, highest impact)
3. **Ensure YOLO runs on GPU** (environment fix)
4. **Use FP16 on L4** (1 line change)
5. **torch.compile with reduce-overhead** (1 line, but budget the compile time)
6. **Cap detections at 100** (1 line)

With items 1-4, estimated total: **~80-120s** for 200 images on L4.
