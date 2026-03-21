"""
Lightning LoRA training: progressive layer dropout for intelligent pruning.

Trains with increasing dropout on deeper layers, forcing early layers to
compensate. After training, measures which layers became truly redundant
and can be surgically removed.

Two phases:
  Phase A (4h): Train with progressive layer dropout schedule
  Phase B (30min): Score layer importance via angular distance + LoRA norms

Output: layer_importance.json with per-layer scores, recommended pruning set.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python train_lightning.py [--resume PATH]

  # Or with torchrun for multi-GPU:
  CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train_lightning.py
"""

import json
import math
import functools
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

print = functools.partial(print, flush=True)

# === CONFIG ===
PRUNED_DIR = Path(__file__).parent / "pruned"
COMP_SAMPLES = Path(__file__).parent / "cached_dataset" / "samples.json"
VAL_DIR = Path(__file__).parent.parent / "data-creation" / "data" / "stratified_split" / "val"
OUTPUT_DIR = Path(__file__).parent / "lightning_output"
CHECKPOINT = Path(__file__).parent / "training_output" / "best" / "best.pt"

NUM_CLASSES = 356
BATCH_SIZE = 8
LR = 5e-5
EPOCHS = 8
WARMUP_STEPS = 200
LOG_EVERY = 10
SAVE_EVERY = 500
MAX_HOURS = 4.0

# LayerSkip dropout schedule: p(l) = p_max * (l / L)^exponent
P_MAX = 0.5       # Max dropout probability (for deepest layer)
EXPONENT = 2.0    # Controls curve shape (2 = quadratic, more aggressive on deep layers)
NUM_TEXT_LAYERS = 12  # In our pruned model


class ClassificationHead(nn.Module):
    def __init__(self, h=1024, nc=356):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, nc))
    def forward(self, x):
        return self.head(x.mean(dim=1))


class CropDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {"image": Image.open(s["crop_path"]).convert("RGB"),
                "label": s["category_id"]}


def get_dropout_prob(layer_idx, num_layers, p_max=P_MAX, exponent=EXPONENT):
    """Progressive dropout: deeper layers get higher dropout."""
    return p_max * (layer_idx / max(1, num_layers - 1)) ** exponent


class LayerDropoutWrapper(nn.Module):
    """Wraps a transformer layer with stochastic depth (layer dropout).

    During training, randomly skips the layer with probability p.
    When skipped, the layer becomes an identity (residual stream passes through).
    """
    def __init__(self, layer, dropout_prob):
        super().__init__()
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.times_skipped = 0
        self.times_used = 0

    def __getattr__(self, name):
        """Proxy attribute access to wrapped layer (e.g. layer_type)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        if self.training and torch.rand(1).item() < self.dropout_prob:
            # Skip this layer: return input unchanged
            self.times_skipped += 1
            # The first positional arg is hidden_states
            return args[0] if len(args) > 0 else kwargs.get('hidden_states')
        else:
            self.times_used += 1
            return self.layer(*args, **kwargs)


def wrap_layers_with_dropout(model):
    """Replace each transformer layer with a LayerDropoutWrapper."""
    # Access the language model layers
    lm = model.model.language_model if hasattr(model.model, 'language_model') else model.model
    layers = lm.layers

    wrappers = []
    for i in range(len(layers)):
        p = get_dropout_prob(i, len(layers))
        wrapper = LayerDropoutWrapper(layers[i], p)
        layers[i] = wrapper
        wrappers.append(wrapper)
        print(f"  Layer {i}: dropout_prob={p:.3f}")

    return wrappers


def compute_layer_importance(model, processor, dataset, device, num_batches=50):
    """Score layer importance using angular distance + skip statistics.

    Returns dict with per-layer scores (lower = more redundant = safe to remove).
    """
    print("\n=== Computing Layer Importance ===")
    model.eval()

    lm = model.model.language_model if hasattr(model.model, 'language_model') else model.model
    layers = lm.layers
    num_layers = len(layers)

    # Collect angular distances via hooks
    angular_distances = torch.zeros(num_layers)
    layer_io = {}

    hooks = []
    for i in range(num_layers):
        layer = layers[i]
        # Get the actual layer (unwrap if wrapped)
        actual = layer.layer if isinstance(layer, LayerDropoutWrapper) else layer

        def make_hook(idx):
            def hook_fn(module, inp, out):
                # inp[0] = hidden states input, out[0] or out = hidden states output
                h_in = inp[0].detach() if isinstance(inp, tuple) else inp.detach()
                h_out = out[0].detach() if isinstance(out, tuple) else out.detach()
                layer_io[idx] = (h_in, h_out)
            return hook_fn
        hooks.append(actual.register_forward_hook(make_hook(i)))

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0,
                        collate_fn=lambda b: {"images": [x["image"] for x in b],
                                              "labels": torch.tensor([x["label"] for x in b])},
                        drop_last=True)

    # Temporarily disable dropout for measurement
    for layer in layers:
        if isinstance(layer, LayerDropoutWrapper):
            layer.dropout_prob = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_batches:
                break
            images = batch["images"]
            msgs_list = []
            for img in images:
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "classify"}]}]
                msgs_list.append(processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False))

            inputs = processor(images=images, text=msgs_list,
                             return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            model.model(**inputs, output_hidden_states=True)

            for i in range(num_layers):
                if i in layer_io:
                    h_in, h_out = layer_io[i]
                    h_in_flat = h_in.flatten(0, 1)
                    h_out_flat = h_out.flatten(0, 1)
                    cos = F.cosine_similarity(h_in_flat, h_out_flat, dim=-1)
                    ang = torch.arccos(cos.clamp(-0.9999, 0.9999)).mean()
                    angular_distances[i] += ang.item()

    angular_distances /= num_batches

    # Remove hooks
    for h in hooks:
        h.remove()

    # Restore dropout probs
    for i, layer in enumerate(layers):
        if isinstance(layer, LayerDropoutWrapper):
            layer.dropout_prob = get_dropout_prob(i, num_layers)

    # Skip statistics from training
    skip_stats = {}
    for i, layer in enumerate(layers):
        if isinstance(layer, LayerDropoutWrapper):
            total = layer.times_used + layer.times_skipped
            skip_rate = layer.times_skipped / max(1, total)
            skip_stats[i] = {
                "times_used": layer.times_used,
                "times_skipped": layer.times_skipped,
                "skip_rate": skip_rate,
            }

    # Combine scores: low angular distance + high skip rate = redundant
    importance = {}
    for i in range(num_layers):
        ang = angular_distances[i].item()
        skip = skip_stats.get(i, {}).get("skip_rate", 0)
        # Importance score: higher = more important
        # Angular distance is HIGH for important layers (they transform a lot)
        # We want to KEEP high-angular-distance layers
        importance[i] = {
            "angular_distance": ang,
            "skip_rate": skip,
            "importance_score": ang * (1 - skip * 0.3),  # Penalize high skip rate
        }

    # Rank layers
    ranked = sorted(importance.items(), key=lambda x: x[1]["importance_score"])

    print("\nLayer rankings (most redundant first):")
    print(f"{'Layer':>6} {'AngDist':>8} {'SkipRate':>9} {'Score':>8} {'Verdict':>10}")
    for i, (layer_idx, info) in enumerate(ranked):
        verdict = "REMOVE" if i < 4 else "KEEP"
        print(f"{layer_idx:>6} {info['angular_distance']:>8.4f} {info['skip_rate']:>9.3f} {info['importance_score']:>8.4f} {verdict:>10}")

    # Recommendation
    removable = [idx for idx, _ in ranked[:4]]
    keepable = [idx for idx, _ in ranked[4:]]

    result = {
        "per_layer": importance,
        "ranking": [(idx, info) for idx, info in ranked],
        "recommended_remove": sorted(removable),
        "recommended_keep": sorted(keepable),
        "original_layers": num_layers,
        "target_layers": num_layers - len(removable),
    }

    return result


def process_batch(images, processor, device):
    texts = []
    for img in images:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "classify"}]}]
        texts.append(processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False))
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        import wandb
        wandb.init(project="nmiai-objdet", name="lightning-lora-dropout",
                   config={"p_max": P_MAX, "exponent": EXPONENT, "lr": LR, "epochs": EPOCHS})
        use_wandb = True
    except Exception:
        use_wandb = False

    # Load model
    print("Loading pruned model...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(PRUNED_DIR), dtype=torch.bfloat16,
        ignore_mismatched_sizes=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    h = model.config.text_config.hidden_size

    cls_head = ClassificationHead(h, NUM_CLASSES).to(device).to(torch.bfloat16)

    # Load trained weights if available
    if CHECKPOINT.exists():
        print(f"Loading checkpoint: {CHECKPOINT}")
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        cls_head.load_state_dict(ckpt["cls_head_state"])
        print(f"Loaded (acc={ckpt.get('accuracy', 0):.3f})")

    model = model.to(device)

    # Wrap layers with progressive dropout
    print("\nApplying progressive layer dropout:")
    wrappers = wrap_layers_with_dropout(model)

    # Data
    with open(COMP_SAMPLES) as f:
        samples = json.load(f)
    dataset = CropDataset(samples)
    print(f"Dataset: {len(dataset)} samples")

    counts = Counter(s["category_id"] for s in samples)
    total_n = sum(counts.values())
    class_weights = torch.zeros(NUM_CLASSES, device=device, dtype=torch.bfloat16)
    for c in range(NUM_CLASSES):
        cnt = counts.get(c, 0)
        class_weights[c] = (total_n / (NUM_CLASSES * cnt)) if cnt > 0 else 1.0
    class_weights = class_weights.clamp(max=10.0)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        collate_fn=lambda b: {"images": [x["image"] for x in b],
                                              "labels": torch.tensor([x["label"] for x in b])},
                        drop_last=True)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS

    all_params = list(model.parameters()) + list(cls_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        min(s / max(1, WARMUP_STEPS), 0.5 * (1 + math.cos(math.pi * max(0, s - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)))))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    cls_head.train()
    global_step = 0
    best_acc = 0
    start_time = time.time()

    print(f"\n=== Phase A: Training with progressive layer dropout ({EPOCHS} epochs) ===\n")

    for epoch in range(EPOCHS):
        el = ec = et = 0
        for batch_idx, batch in enumerate(loader):
            elapsed = (time.time() - start_time) / 3600
            if elapsed >= MAX_HOURS:
                print(f"Time limit ({MAX_HOURS}h) reached")
                break

            images = batch["images"]
            labels = batch["labels"].to(device)
            inputs = process_batch(images, processor, device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.model(**inputs, output_hidden_states=True)
                logits = cls_head(outputs.last_hidden_state)
                loss = F.cross_entropy(logits, labels, weight=class_weights,
                                       label_smoothing=0.05)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            preds = logits.argmax(-1)
            ec += (preds == labels).sum().item()
            et += labels.shape[0]
            el += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                acc = ec / max(1, et)
                lr = scheduler.get_last_lr()[0]
                # Show skip stats
                skip_info = " ".join(f"L{i}:{w.times_skipped}/{w.times_used+w.times_skipped}"
                                     for i, w in enumerate(wrappers) if (w.times_used + w.times_skipped) > 0)
                print(f"Step {global_step}/{total_steps} | loss={loss.item():.4f} avg={el/(batch_idx+1):.4f} | acc={acc:.3f} | {elapsed:.1f}h")
                if use_wandb:
                    log = {"lightning/loss": loss.item(), "lightning/accuracy": acc, "lightning/lr": lr}
                    for i, w in enumerate(wrappers):
                        t = w.times_used + w.times_skipped
                        if t > 0:
                            log[f"lightning/skip_rate_L{i}"] = w.times_skipped / t
                    wandb.log(log, step=global_step)

            if global_step % SAVE_EVERY == 0:
                acc = ec / max(1, et)
                if acc > best_acc:
                    best_acc = acc
                    torch.save({
                        "model_state": model.state_dict(),
                        "cls_head_state": cls_head.state_dict(),
                        "step": global_step, "accuracy": acc,
                    }, OUTPUT_DIR / "best.pt")

        acc = ec / max(1, et)
        print(f"\nEpoch {epoch+1}/{EPOCHS}: loss={el/max(1,batch_idx+1):.4f} acc={acc:.3f}\n")
        elapsed = (time.time() - start_time) / 3600
        if elapsed >= MAX_HOURS:
            break

    # Phase B: Score layer importance
    print("\n=== Phase B: Scoring layer importance ===")
    importance = compute_layer_importance(model, processor, dataset, device)

    # Save results
    # Convert non-serializable items
    serializable = {
        "recommended_remove": importance["recommended_remove"],
        "recommended_keep": importance["recommended_keep"],
        "original_layers": importance["original_layers"],
        "target_layers": importance["target_layers"],
        "per_layer": {str(k): {kk: float(vv) for kk, vv in v.items()}
                      for k, v in importance["per_layer"].items()},
    }
    with open(OUTPUT_DIR / "layer_importance.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved layer_importance.json to {OUTPUT_DIR}")
    print(f"Recommended layers to REMOVE: {importance['recommended_remove']}")
    print(f"Recommended layers to KEEP: {importance['recommended_keep']}")
    print(f"Target: {importance['original_layers']} -> {importance['target_layers']} layers")

    # Final save
    torch.save({
        "model_state": model.state_dict(),
        "cls_head_state": cls_head.state_dict(),
        "layer_importance": serializable,
        "step": global_step, "accuracy": best_acc,
    }, OUTPUT_DIR / "final.pt")

    if use_wandb:
        wandb.finish()
    print("\nLIGHTNING TRAINING COMPLETE")


if __name__ == "__main__":
    train()
