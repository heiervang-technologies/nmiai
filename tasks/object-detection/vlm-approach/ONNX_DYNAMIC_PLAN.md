# MarkusNet Vision Encoder ONNX Dynamic Plan

## Context

Problem: MarkusNet classification drops from about 90.8% to 56.9% when the Qwen3.5-VLM vision encoder is exported with a fixed square input and crops are distorted at inference.

The key fact is that Qwen3.5-VLM does **not** naturally consume `B x C x H x W` images in the vision tower. The processor converts each image into:

- `pixel_values`: shape `(num_patches, 1536)`
- `image_grid_thw`: shape `(num_images, 3)`

For single-image classification, `image_grid_thw` is typically `[[1, grid_h, grid_w]]`, and the number of image tokens inserted into the prompt is:

- `num_image_tokens = (grid_t * grid_h * grid_w) / merge_size^2`

Observed from the actual processor:

- `(160, 320)` -> `pixel_values (288, 1536)`, `grid [[1, 24, 12]]`, `72` image tokens
- `(256, 256)` -> `pixel_values (256, 1536)`, `grid [[1, 16, 16]]`, `64` image tokens
- `(128, 384)` -> `pixel_values (280, 1536)`, `grid [[1, 28, 10]]`, `70` image tokens
- `(448, 448)` -> `pixel_values (784, 1536)`, `grid [[1, 28, 28]]`, `196` image tokens

So the real variable dimension is not just image `H/W`; it is the patch sequence length plus the associated `grid_thw` metadata.

## What The HF/Qwen Stack Is Doing

The processor uses `smart_resize(...)` from the Qwen2/Qwen3 VL image processor family.

- It preserves aspect ratio.
- It resizes so both dimensions are divisible by `patch_size * merge_size`.
- For this model, the processor uses:
  - `patch_size = 14`
  - `temporal_patch_size = 2`
  - `merge_size = 2`

The pruned MarkusNet config on disk says `vision_config.patch_size = 16` and `spatial_merge_size = 2`, but the actual upstream processor feeding Qwen3.5-VLM is generating `pixel_values` with patch width `14`, yielding patch vectors of `3 * 2 * 14 * 14 = 1176` for standard Qwen2-VL, while our observed MarkusNet path is using `1536`, which corresponds to `3 * 2 * 16 * 16`. The important operational point is:

- **Do not export a raw `B x C x H x W` image interface unless you also faithfully reimplement Qwen preprocessing.**
- The safest ONNX contract is the same one the vision tower actually consumes: `pixel_values` plus `image_grid_thw`.

## Option 1: `torch.onnx.export` With Dynamic Spatial Axes

### Verdict

Not the recommended primary path.

### Why

Exporting with dynamic `H/W` only works cleanly when the model forward is naturally expressed as image tensors with interpolation logic inside the graph.

Qwen3.5-VLM vision is different:

- the processor converts images into flattened patch rows before the vision tower runs
- the token count depends on resized image dimensions
- the prompt token count must also match `grid_thw`

If you export a wrapper taking raw image tensors, you must reproduce:

- smart resize
- normalization
- temporal duplication
- patch flattening layout
- grid computation

inside the ONNX graph.

That is possible in theory, but much higher risk under deadline.

### Risk

- High exporter complexity
- High chance of mismatch with training-time preprocessing
- Harder to validate quickly

## Option 2: Dynamic `grid_thw` ONNX Export

### Verdict

This is the recommended path.

### Why

The HF vision forward already takes:

- `hidden_states` / `pixel_values`
- `grid_thw`

and computes position embeddings and rotary embeddings from `grid_thw` dynamically.

Relevant behavior in the vision model:

- `fast_pos_embed_interpolate(grid_thw)` interpolates learned position embeddings based on runtime grid size
- `rot_pos_emb(grid_thw)` builds rotary embeddings from runtime grid size
- `cu_seqlens` is derived from `grid_thw[:,1] * grid_thw[:,2]`

This means the model architecture itself is already designed for variable-size patch sequences. The dynamic part is first-class in the forward, not an afterthought.

### Recommended ONNX contract

Export only the MarkusNet vision encoder with inputs:

- `pixel_values`: `(num_patches, patch_dim)`
- `image_grid_thw`: `(1, 3)`

and output:

- `vision_embeds`: `(num_merged_tokens, 1024)` or whatever your wrapper consumes next

### Dynamic axes recommendation

Use dynamic axes for:

- `pixel_values[0]` = `num_patches`
- `image_grid_thw[0]` = `num_images` (can stay fixed at 1 for crop classification if you want)
- `vision_embeds[0]` = `num_image_tokens`

Example shape policy:

```python
dynamic_axes = {
    "pixel_values": {0: "num_patches"},
    "image_grid_thw": {0: "num_images"},
    "vision_embeds": {0: "num_image_tokens"},
}
```

Do **not** bother making the feature width dynamic.

### Practical export wrapper

Use a minimal wrapper around `model.visual` or your local `VisionEncoder` equivalent:

```python
class VisionOnlyWrapper(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values, image_grid_thw):
        out = self.visual(pixel_values, grid_thw=image_grid_thw)
        return out.pooler_output if hasattr(out, "pooler_output") else out
```
```

Then export with a representative non-square sample.

### Expected caveats

- Some ops in Qwen attention stacks can be awkward in ONNX if you export the full multimodal model. Exporting the **vision tower only** is much safer.
- If HF exporter has issues with the stock module, use the already-local pure PyTorch `VisionEncoder` implementation in `run_markusnet.py` as the export target instead. That code is much simpler and already mirrors your inference path.

### Why this is the best deadline tradeoff

- preserves aspect ratio behavior
- preserves variable token count
- avoids square distortion
- avoids exporting the language model together with multimodal prompt plumbing
- matches how the classifier already reasons about image tokens

## Option 3: Fixed Square Export With Letterbox Padding

### Verdict

Best fallback if dynamic ONNX export becomes unstable or blocked.

### Why it is much better than naive resize

The current accuracy collapse is caused by **distortion**, not merely fixed size.

If you export at a fixed square size and preprocess with **letterbox padding** instead of aspect-ratio-destroying resize, you preserve object geometry.

That means:

- resize crop isotropically so the longer side fits target
- pad the remaining area with a constant color
- normalize exactly as in training/inference

This will still differ from Qwen’s native dynamic patch grid behavior, but it is much closer than brute-force square stretch.

### Recommended fixed-size fallback

- Try `256 x 256` first for cost/speed
- Also benchmark `320 x 320` or `384 x 384` if budget allows
- Use letterbox, not plain resize

### Risks

- still changes token/grid geometry compared with native dynamic path
- may leave some accuracy on the table versus true dynamic export

## Final Recommendation

### Primary recommendation

Export the **vision encoder only** to ONNX with dynamic sequence length using:

- `pixel_values` as dynamic input
- `image_grid_thw` as dynamic metadata input

This is the path most aligned with the real Qwen3.5-VLM design.

### Secondary fallback

If exporter issues block that path, export a fixed-size vision model and use **letterbox padding to square**, not stretch-resize.

### Do not recommend

- Exporting raw `B x C x H x W` dynamic image tensors as the primary plan
- Naive fixed square resize without padding

## Concrete Action Plan

1. Build a `VisionOnlyWrapper` around the vision tower.
2. Export ONNX with inputs `pixel_values` and `image_grid_thw`.
3. Mark `num_patches` dynamic.
4. Validate ONNX output numerically against PyTorch on at least 3 shapes:
   - tall crop
   - square crop
   - wide crop
5. Run classification validation on the same crop set used for the 90.8% measurement.
6. If ONNX export/runtime fails, switch immediately to fixed `256x256` letterbox export and re-benchmark.

## Suggested Decision Under Time Pressure

If the team needs a go/no-go call now:

- **Go with dynamic `pixel_values + image_grid_thw` export first.**
- **Keep letterboxed fixed-square ONNX as the backup plan.**
- **Do not spend time on raw dynamic `H/W` image export unless the first path is impossible.**

