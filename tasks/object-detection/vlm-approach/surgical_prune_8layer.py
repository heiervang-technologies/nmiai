"""Surgical pruning: remove layers 3, 6, 7, 11 from 12-layer MarkusNet.

Based on lightning LoRA layer importance analysis:
- L3 (full_attention): angular_dist=0.139, importance=0.139
- L6 (linear_attention): angular_dist=0.145, importance=0.142
- L7 (full_attention): angular_dist=0.116, importance=0.113
- L11 (full_attention): angular_dist=0.120, importance=0.112

Result: 8 layers, all linear_attention. ~67% size of 12-layer.
"""
import json
import copy
import functools
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import save_file, load_file

print = functools.partial(print, flush=True)

PRUNED_DIR = Path(__file__).parent / "pruned"
OUTPUT_DIR = Path(__file__).parent / "pruned_8layer_surgical"
CHECKPOINT = Path(__file__).parent / "training_output_multitask" / "best" / "best.pt"

REMOVE_LAYERS = [3, 6, 7, 11]
KEEP_LAYERS = [0, 1, 2, 4, 5, 8, 9, 10]


def main():
    print("=== Surgical Pruning: 12 -> 8 layers ===")
    print(f"Removing layers: {REMOVE_LAYERS}")
    print(f"Keeping layers: {KEEP_LAYERS}")

    # Load config
    config_path = PRUNED_DIR / "config.json"
    config = json.load(open(config_path))

    old_layer_types = config["text_config"]["layer_types"]
    print(f"\nOriginal layer types: {old_layer_types}")

    new_layer_types = [old_layer_types[i] for i in KEEP_LAYERS]
    print(f"New layer types: {new_layer_types}")

    # Update config
    new_config = copy.deepcopy(config)
    new_config["text_config"]["num_hidden_layers"] = len(KEEP_LAYERS)
    new_config["text_config"]["layer_types"] = new_layer_types

    # Update full_attention_interval if needed
    # With all linear_attention layers, set interval to something > num_layers
    new_config["text_config"]["full_attention_interval"] = 999

    # Load checkpoint weights
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    cls_state = ckpt["cls_head_state"]

    # Build layer index mapping: old_idx -> new_idx
    layer_map = {old_idx: new_idx for new_idx, old_idx in enumerate(KEEP_LAYERS)}
    print(f"Layer mapping: {layer_map}")

    # Remap state dict keys
    new_state = OrderedDict()
    skipped_params = 0
    kept_params = 0

    for key, value in state.items():
        # Check if this key belongs to a language model layer
        # Pattern: model.language_model.layers.N.xxx
        parts = key.split(".")
        is_layer_param = False

        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                if layer_idx in REMOVE_LAYERS:
                    skipped_params += value.numel()
                    is_layer_param = True
                    break
                elif layer_idx in layer_map:
                    # Remap layer index
                    new_parts = list(parts)
                    new_parts[i + 1] = str(layer_map[layer_idx])
                    new_key = ".".join(new_parts)
                    new_state[new_key] = value
                    kept_params += value.numel()
                    is_layer_param = True
                    break

        if not is_layer_param:
            # Non-layer params (embeddings, vision, norms, etc.) - keep as-is
            new_state[key] = value
            kept_params += value.numel()

    print(f"\nKept params: {kept_params/1e6:.1f}M")
    print(f"Removed params: {skipped_params/1e6:.1f}M")
    print(f"Reduction: {skipped_params/(kept_params+skipped_params)*100:.1f}%")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)
    print(f"\nSaved config to {OUTPUT_DIR / 'config.json'}")

    # Save model weights as safetensors
    # Convert all to float16 for consistency
    st_state = {k: v.to(torch.float16) if v.is_floating_point() else v for k, v in new_state.items()}
    save_file(st_state, str(OUTPUT_DIR / "model.safetensors"))
    print(f"Saved model to {OUTPUT_DIR / 'model.safetensors'}")

    # Save cls head separately
    torch.save({"cls_head_state": cls_state}, OUTPUT_DIR / "cls_head.pt")
    print(f"Saved cls_head to {OUTPUT_DIR / 'cls_head.pt'}")

    # Also save as a combined checkpoint for training
    torch.save({
        "model_state": new_state,
        "cls_head_state": cls_state,
        "pruned_from": "12-layer surgical",
        "removed_layers": REMOVE_LAYERS,
        "kept_layers": KEEP_LAYERS,
    }, OUTPUT_DIR / "checkpoint.pt")

    # Size report
    safetensors_size = (OUTPUT_DIR / "model.safetensors").stat().st_size / 1024**2
    print(f"\nModel size: {safetensors_size:.1f} MB (FP16)")
    print(f"Estimated NF4: {safetensors_size * 180/667:.1f} MB")
    print(f"With YOLO (132 MB): {safetensors_size * 180/667 + 132:.1f} MB total")

    # Verify layer count
    layer_keys = set()
    for k in new_state.keys():
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_keys.add(int(parts[i + 1]))
    print(f"\nLayer indices in new state: {sorted(layer_keys)}")
    print(f"Expected: {list(range(len(KEEP_LAYERS)))}")

    print("\n=== SURGICAL PRUNING COMPLETE ===")


if __name__ == "__main__":
    main()
