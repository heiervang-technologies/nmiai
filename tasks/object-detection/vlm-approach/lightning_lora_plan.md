# Lightning LoRA Pruning: Implementation Plan

## Executive Summary

Lightning LoRA Pruning combines ideas from LayerSkip (Meta FAIR, ACL 2024), ShortGPT / "The Unreasonable Ineffectiveness of the Deeper Layers" (ICLR 2025), and LoRA-drop to intelligently prune transformer layers from Qwen3.5-0.8B. Instead of naively dropping the last N layers, we train with progressive layer dropout so the model learns to be robust to missing layers, then surgically remove the least important ones — potentially from the middle of the network, not just the tail.

**Target**: Match or exceed 91% classification accuracy (currently from 12 of 24 text blocks) using only 8 text blocks (~60MB savings in INT8).

---

## 1. Literature Review

### 1.1 LayerSkip (Meta FAIR, ACL 2024)

**Paper**: [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)
**Code**: [github.com/facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)

Key contributions relevant to us:
- **Progressive layer dropout during training**: Low dropout on early layers, high dropout on later layers. This forces the model to produce useful representations at every depth.
- **Dropout schedule**: Layer `l` out of `L` total layers gets dropout probability `p_l = p_max * (l / L)^e` where `e` controls the curve shape (linear when e=1, more aggressive when e>1). The paper uses `p_max` up to 0.5.
- **Early exit loss**: All layers share the same LM head / exit, with a weighted sum of losses from each layer's output. The scale factor `e_scale` down-weights early layer losses.
- **Results**: Up to 2.16x speedup with minimal quality loss. Key insight: after training with layer dropout, early/middle layers become much more capable of producing final-quality outputs.

### 1.2 The Unreasonable Ineffectiveness of the Deeper Layers (ICLR 2025)

**Paper**: [arxiv.org/abs/2403.17887](https://arxiv.org/abs/2403.17887)
**Code**: [github.com/melisa-writer/short-transformers](https://github.com/melisa-writer/short-transformers)

Key findings:
- **Up to half the layers can be removed** from large models (Llama-2-70B) with minimal degradation on QA benchmarks.
- **Angular distance metric**: Measures cosine similarity between hidden states before and after each layer. Layers with small angular distance (output ~= input) are candidates for removal.
- **Deeper layers are most redundant**: The angular distance between consecutive layers decreases with depth, meaning deeper layers transform their inputs less.
- **Healing via QLoRA**: After pruning, a small amount of finetuning (they call it "healing") with QLoRA on a single A100 recovers most of the lost performance.
- **Strategy**: Drop deepest layers (excluding the very final layer before the LM head), then heal.

### 1.3 The Curse of Depth (arxiv 2502.05795)

**Paper**: [arxiv.org/abs/2502.05795](https://arxiv.org/abs/2502.05795)

Explains WHY deeper layers are ineffective:
- **Pre-LayerNorm causes output variance to grow exponentially** with depth.
- Deep layers' gradients become identity matrices — they barely learn anything during training.
- The Qwen model family is explicitly listed as affected.
- **Proposed fix**: LayerNorm Scaling (LNS) — scale normalization variance inversely by sqrt(depth). This is relevant because it means our pruned model could benefit from adjusting LayerNorm scales on the surviving layers.

### 1.4 SWIFT (ICLR 2025)

**Paper**: [openreview.net/forum?id=EKJhH5D5wA](https://openreview.net/forum?id=EKJhH5D5wA)

- On-the-fly self-speculative decoding that adaptively selects which layers to skip.
- Key insight: optimal skip sets are **input-dependent** — different inputs benefit from different layer subsets.
- Larger models have greater layer sparsity (more layers can be skipped).
- Less directly useful for us (inference-only, no training), but confirms that non-contiguous layer skipping is viable.

### 1.5 LoRA-drop (COLING 2025)

**Paper**: [arxiv.org/abs/2402.07721](https://arxiv.org/abs/2402.07721)

- Evaluates LoRA importance by measuring the output magnitude of each LoRA adapter.
- Retains LoRA for important layers, shares a single LoRA for unimportant layers.
- Achieves comparable performance with 50% fewer LoRA parameters.
- **Useful metric**: The LoRA output norm `||BA * x||` directly measures how much each layer's adaptation matters.

---

## 2. Which Layers Are Redundant?

### 2.1 General Pattern (from literature)

For standard Pre-LN transformers:
- **Layers 0-2 (early)**: Critical. Handle tokenization residuals, positional encoding integration, and basic feature extraction. Never prune these.
- **Layers 3 to ~60% depth (middle)**: Show high uniformity but are NOT truly redundant. They refine representations incrementally.
- **Layers ~60% to N-1 (late)**: Most redundant per angular distance. The Curse of Depth explains why — Pre-LN causes their gradients to vanish.
- **Layer N-1 (final)**: Often important — it's the interface to the output head. Keep it if possible.

### 2.2 For Qwen3.5-0.8B Specifically

Architecture: 24 text transformer blocks + 12 vision blocks (ViT) + merger.

Currently pruned to 12 text blocks (0-11), dropping 12-23. This is a naive "keep first half" approach.

**Prediction based on literature**: With intelligent pruning, we should be able to identify 4 layers in the 0-11 range that contribute least (likely in the 5-9 range — the "uniform middle") and remove them, keeping 8 layers total.

### 2.3 Angular Distance for VLMs

The angular distance metric from ShortGPT was designed for text-only LLMs. For VLMs:
- The vision encoder processes a fundamentally different modality and should NOT be pruned using the same metric.
- The text decoder layers that immediately follow the vision-language merger (projector) are likely critical for cross-modal integration.
- We should only apply angular distance scoring to text decoder layers, and weight the first 2-3 post-merger layers as "protected."

---

## 3. Implementation Plan

### Phase 1: Importance Scoring (No Training Required)

**Goal**: Measure which of the 24 text blocks are most/least important before any training.

```python
# importance_scoring.py — pseudocode

import torch
from transformers import AutoModelForImageTextToText

def compute_angular_distances(model, dataloader, num_layers=24):
    """
    For each layer l, measure how much it changes its input.
    angular_distance[l] = mean(arccos(cos_sim(h_l, h_{l-1}))) over dataset

    Low angular distance = layer barely transforms input = candidate for removal.
    """
    angular_distances = torch.zeros(num_layers)

    # Hook into each layer's input and output
    hooks = []
    layer_inputs = {}
    layer_outputs = {}

    for l, layer in enumerate(model.model.layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # input[0] is the hidden state
                layer_inputs[layer_idx] = input[0].detach()
                # output[0] is the hidden state after the layer
                layer_outputs[layer_idx] = output[0].detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(l)))

    with torch.no_grad():
        for batch in dataloader:
            model(**batch)
            for l in range(num_layers):
                inp = layer_inputs[l].flatten(0, 1)  # [B*seq, hidden]
                out = layer_outputs[l].flatten(0, 1)
                cos = F.cosine_similarity(inp, out, dim=-1)
                angular_distances[l] += torch.arccos(cos.clamp(-1, 1)).mean()

    angular_distances /= len(dataloader)

    for h in hooks:
        h.remove()

    return angular_distances  # Low = redundant
```

**Also measure**: LoRA-drop style output norms if LoRA is already attached.

### Phase 2: Progressive Layer Dropout Training

**Goal**: Train a LoRA adapter with progressive dropout so early/middle layers learn to compensate for dropped later layers.

```python
# progressive_dropout_training.py — pseudocode

class ProgressiveLayerDropout(torch.nn.Module):
    """
    Wraps a transformer and applies progressive layer dropout during training.
    Based on LayerSkip (Meta FAIR, ACL 2024).
    """

    def __init__(self, model, p_max=0.4, exponent=1.0, protected_layers=None):
        super().__init__()
        self.model = model
        self.p_max = p_max          # Max dropout prob (for deepest layer)
        self.exponent = exponent     # 1.0=linear, 2.0=quadratic ramp
        self.protected_layers = protected_layers or {0, 1}  # Never drop these
        self.num_layers = len(model.model.layers)

    def get_dropout_prob(self, layer_idx):
        """Progressive dropout: p(l) = p_max * (l / L)^exponent"""
        if layer_idx in self.protected_layers:
            return 0.0
        return self.p_max * (layer_idx / (self.num_layers - 1)) ** self.exponent

    def forward(self, *args, **kwargs):
        if not self.training:
            return self.model(*args, **kwargs)

        # Monkey-patch each layer's forward to include stochastic skip
        original_forwards = {}
        for l, layer in enumerate(self.model.model.layers):
            p = self.get_dropout_prob(l)
            if p > 0 and torch.rand(1).item() < p:
                # Skip this layer: replace forward with identity
                original_forwards[l] = layer.forward
                layer.forward = lambda *a, **kw: a[0]  # Return input unchanged

        output = self.model(*args, **kwargs)

        # Restore original forwards
        for l, fwd in original_forwards.items():
            self.model.model.layers[l].forward = fwd

        return output
```

**Training recipe**:
1. Load Qwen3.5-0.8B (all 24 text layers)
2. Attach LoRA adapters (rank 32) to all attention + MLP modules
3. Wrap in `ProgressiveLayerDropout(p_max=0.4, exponent=1.5)`
4. Train on grocery classification crops for 3 epochs
5. Use higher learning rate than normal (2x-3x) per LayerSkip findings
6. Early exit loss (optional): Add auxiliary classification loss at layers 8, 12, 16

**Hyperparameters** (from LayerSkip paper):
- `p_max`: 0.3-0.5 (start conservative at 0.3)
- `exponent`: 1.0-2.0 (1.5 is a good default — drops later layers much more)
- Learning rate: 3e-4 to 5e-4 (higher than standard LoRA fine-tuning)
- Protected layers: {0, 1, 23} (first two + final layer)

### Phase 3: Post-Training Importance Measurement

After training with progressive dropout, re-measure importance:

```python
def measure_post_training_importance(model, dataloader, num_layers=24):
    """
    Three complementary metrics:

    1. Angular distance (ShortGPT): How much does each layer change its input?
    2. LoRA output norm (LoRA-drop): How much does the LoRA adapter contribute?
    3. Ablation sensitivity: What happens to loss when each layer is individually skipped?
    """

    # Metric 1: Angular distance (see Phase 1)
    angular = compute_angular_distances(model, dataloader, num_layers)

    # Metric 2: LoRA output norms
    lora_norms = torch.zeros(num_layers)
    for l, layer in enumerate(model.model.layers):
        for name, module in layer.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # LoRA output = B @ A @ x, measure ||B @ A|| as proxy
                A = module.lora_A.default.weight  # [r, in]
                B = module.lora_B.default.weight  # [out, r]
                lora_norms[l] += (B @ A).norm().item()

    # Metric 3: Leave-one-out ablation
    ablation_loss = torch.zeros(num_layers)
    baseline_loss = evaluate(model, dataloader)
    for l in range(num_layers):
        # Temporarily replace layer with identity
        orig_forward = model.model.layers[l].forward
        model.model.layers[l].forward = lambda *a, **kw: a[0]
        ablation_loss[l] = evaluate(model, dataloader) - baseline_loss
        model.model.layers[l].forward = orig_forward

    # Composite score: layers with LOW angular distance, LOW lora norm,
    # and LOW ablation sensitivity are safe to remove
    importance = (
        0.3 * normalize(angular) +
        0.3 * normalize(lora_norms) +
        0.4 * normalize(ablation_loss)
    )

    return importance  # Low score = safe to remove
```

### Phase 4: Surgical Layer Removal

```python
def remove_layers(model, layers_to_remove):
    """
    Remove specific layers and patch residual connections.

    In a Pre-LN transformer, each layer is:
        h_{l+1} = h_l + Layer_l(LayerNorm(h_l))

    Removing a layer just means the residual stream skips it.
    Since it's a residual connection, no patching is needed —
    we simply delete the layer from the ModuleList.

    The only thing to fix: positional encoding / RoPE indices
    if they're layer-dependent (they're not in Qwen — RoPE is
    applied per-head based on sequence position, not layer index).
    """
    import copy

    # Sort in reverse so indices don't shift during removal
    to_remove = sorted(layers_to_remove, reverse=True)

    new_layers = torch.nn.ModuleList()
    for i, layer in enumerate(model.model.layers):
        if i not in layers_to_remove:
            new_layers.append(layer)

    model.model.layers = new_layers

    # Update config
    model.config.text_config.num_hidden_layers = len(new_layers)

    return model
```

**Key insight**: In residual-stream architectures (which Qwen is), removing a layer is trivial — you just delete it from the `nn.ModuleList`. The residual connection `h + f(h)` becomes just `h` for that layer. No rewiring needed.

### Phase 5: Merge LoRA and Export

```python
def merge_and_export(model, output_path):
    """
    1. Merge LoRA weights into base weights
    2. Quantize to INT8
    3. Save for submission
    """
    # Step 1: Merge LoRA
    # If using PEFT:
    model = model.merge_and_unload()

    # Step 2: Save merged model
    model.save_pretrained(output_path)

    # Step 3: Quantize to INT8 (for submission size)
    # Use torch.ao.quantization or export to ONNX with INT8
```

**Order matters**: Merge LoRA BEFORE removing layers, because the LoRA weights are indexed by layer number. After merging, the LoRA information is baked into the base weights and layer removal is clean.

Wait — actually the correct order is:
1. Train with progressive dropout + LoRA
2. Merge LoRA into base weights (`merge_and_unload`)
3. Measure importance on the merged model
4. Remove layers from the merged model
5. "Heal" with a short round of additional fine-tuning (new LoRA, ~100 steps)
6. Merge the healing LoRA
7. Quantize and export

---

## 4. Feasibility Estimate: 8 Layers at 91% Accuracy

### Current Baseline
- 12 text layers (naive first-half) = 91% accuracy on 356-class grocery classification
- ~180MB INT8 for text blocks (each layer ~15MB INT8)

### What Literature Suggests
- ShortGPT: Up to 50% of layers removable with minimal degradation in large models. For small models (0.8B), the budget is tighter — maybe 25-33%.
- LayerSkip: Progressive dropout training makes early exit at ~50% depth viable.
- Curse of Depth: Qwen specifically suffers from deep-layer ineffectiveness.

### Optimistic Estimate
Keeping 8 of 24 layers (33%) with intelligent selection:
- 4 layers saved x ~15MB = **~60MB savings**
- With progressive dropout training, the kept layers compensate for removed ones
- **Expected accuracy: 88-92%** (possible to match 91% if the right layers are selected)

### Conservative Estimate
- 10 layers might be the safe target (remove 2 from the current 12)
- 8 layers is ambitious but plausible given:
  - Our task is classification (not generation — simpler)
  - 356 classes is finite (the model doesn't need deep reasoning)
  - Vision features carry most of the signal (text layers just need to map features to labels)

### Risk Assessment
- **High confidence**: We can get to 10 layers (save 30MB) with no accuracy loss
- **Medium confidence**: 8 layers matching 91% (save 60MB) with progressive dropout
- **Low confidence**: Below 8 layers — likely hits a cliff for classification accuracy

---

## 5. Step-by-Step Execution Plan

### Step 1: Quick Importance Scan (1 hour)
- Load full 24-layer Qwen3.5-0.8B
- Run angular distance measurement on 500 classification samples
- Generate a heatmap of layer importance
- Identify candidate layers for removal

### Step 2: Naive Ablation Baseline (1 hour)
- Test removing individual layers from the current 12-layer model
- For each of layers 2-10, skip it and measure accuracy
- This gives us ground truth on which layers matter most for our specific task

### Step 3: Progressive Dropout Training (3-4 hours)
- Implement `ProgressiveLayerDropout` wrapper
- Train LoRA (rank 32) on all 24 layers with `p_max=0.4`
- Monitor per-epoch accuracy at different early-exit points (layers 8, 12, 16)

### Step 4: Post-Training Pruning (1 hour)
- Merge LoRA
- Re-measure all three importance metrics
- Select the best 8-layer subset
- Remove layers and measure accuracy

### Step 5: Healing Fine-tune (1 hour)
- Attach fresh LoRA (rank 16) to the 8-layer model
- Train for 100-200 steps to "heal" the damage
- Merge and quantize to INT8
- Measure final accuracy and file size

### Total estimated time: 6-8 hours

---

## 6. Alternative / Simpler Approaches

If full Lightning LoRA is too complex for the competition timeline:

### 6a. ShortGPT-Only (2 hours)
Skip the progressive dropout training entirely:
1. Measure angular distance on current 12-layer model
2. Remove the 4 layers with lowest angular distance
3. Heal with 100 steps of QLoRA
4. This is simpler but won't benefit from dropout-trained robustness

### 6b. Greedy Ablation (3 hours)
1. Start with 12 layers
2. Iteratively remove the layer whose absence causes the least accuracy drop
3. Stop when accuracy drops below 90%
4. Heal remaining model

### 6c. LayerNorm Scaling Fix (30 minutes add-on)
From the Curse of Depth paper: scale each layer's LayerNorm by `1/sqrt(l)`. This can be applied to any pruned model as a post-hoc fix to make the surviving deep layers more effective. Zero training cost.

---

## Sources

- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding (ACL 2024)](https://arxiv.org/abs/2404.16710)
- [The Unreasonable Ineffectiveness of the Deeper Layers (ICLR 2025)](https://arxiv.org/abs/2403.17887)
- [The Curse of Depth in Large Language Models](https://arxiv.org/abs/2502.05795)
- [SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration (ICLR 2025)](https://openreview.net/forum?id=EKJhH5D5wA)
- [LoRA-drop: Efficient LoRA Parameter Pruning based on Output Evaluation](https://arxiv.org/abs/2402.07721)
- [LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning (ACL 2024 Findings)](https://aclanthology.org/2024.findings-acl.178/)
- [Transformer Layers as Painters](https://arxiv.org/abs/2407.09298)
- [LayerSkip GitHub repository](https://github.com/facebookresearch/LayerSkip)
- [ShortGPT / short-transformers GitHub](https://github.com/melisa-writer/short-transformers)
