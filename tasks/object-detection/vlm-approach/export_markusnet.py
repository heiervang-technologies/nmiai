"""
Export MarkusNet-860M for submission.

Takes the best training checkpoint and:
1. Strips embed_tokens and lm_head (508M params, ~970MB FP16)
2. Keeps: vision encoder + merger + 12 transformer blocks + classification head
3. Quantizes to INT8
4. Saves as a single file ready for submission

Result: 858M -> 351M params, ~335MB INT8

Usage: uv run python export_markusnet.py
"""

import functools
from pathlib import Path

import torch
import torch.nn as nn
import torch.quantization

print = functools.partial(print, flush=True)

CHECKPOINT = Path(__file__).parent / "training_output" / "best" / "best.pt"
OUTPUT_DIR = Path(__file__).parent / "exported"


def export():
    print("=== MarkusNet Export Pipeline ===")
    print(f"Checkpoint: {CHECKPOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    model_state = ckpt["model_state"]
    cls_state = ckpt["cls_head_state"]
    accuracy = ckpt.get("accuracy", "unknown")
    step = ckpt.get("global_step", "unknown")
    print(f"Checkpoint: step={step}, accuracy={accuracy}")

    # Strip embed_tokens and lm_head
    print("\nStripping text embeddings...")
    stripped_state = {}
    stripped_params = 0
    kept_params = 0

    for k, v in model_state.items():
        if "embed_tokens" in k or "lm_head" in k:
            stripped_params += v.numel()
            print(f"  REMOVED: {k} ({v.numel()/1e6:.1f}M params)")
        else:
            stripped_state[k] = v
            kept_params += v.numel()

    cls_params = sum(v.numel() for v in cls_state.values())

    print(f"\nStripped: {stripped_params/1e6:.0f}M params ({stripped_params*2/1024**2:.0f}MB FP16)")
    print(f"Kept: {kept_params/1e6:.0f}M + cls {cls_params/1e6:.1f}M = {(kept_params+cls_params)/1e6:.0f}M params")

    # Save FP16 (for reference / further training)
    fp16_path = OUTPUT_DIR / "markusnet_351m_fp16.pt"
    torch.save({
        "model_state": stripped_state,
        "cls_head_state": cls_state,
        "accuracy": accuracy,
        "global_step": step,
        "architecture": {
            "base": "Qwen3.5-0.8B",
            "text_layers_kept": 12,
            "text_layers_total": 24,
            "hidden_size": 1024,
            "vision_layers": 12,
            "vision_hidden": 768,
            "num_classes": 356,
            "stripped": ["embed_tokens", "lm_head"],
        },
    }, fp16_path)
    fp16_size = fp16_path.stat().st_size / 1024**2
    print(f"\nFP16 saved: {fp16_path} ({fp16_size:.0f}MB)")

    # Quantize to INT8
    # For state dict quantization, convert float tensors to int8 + scale
    print("\nQuantizing to INT8...")
    int8_state = {}
    scales = {}

    for k, v in stripped_state.items():
        if v.dtype in (torch.float32, torch.float16, torch.bfloat16) and v.dim() >= 2:
            # Per-channel quantization for weight matrices
            v_float = v.float()
            max_val = v_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = max_val / 127.0
            v_int8 = (v_float / scale).round().clamp(-128, 127).to(torch.int8)
            int8_state[k] = v_int8
            scales[k] = scale.squeeze(-1).to(torch.float16)
        else:
            # Keep small tensors (norms, biases) as-is in float16
            int8_state[k] = v.to(torch.float16) if v.is_floating_point() else v

    # Classification head stays FP16 (tiny)
    cls_fp16 = {k: v.to(torch.float16) for k, v in cls_state.items()}

    int8_path = OUTPUT_DIR / "markusnet_351m_int8.pt"
    torch.save({
        "model_state": int8_state,
        "scales": scales,
        "cls_head_state": cls_fp16,
        "accuracy": accuracy,
        "global_step": step,
        "quantization": "per_channel_int8",
        "architecture": {
            "base": "Qwen3.5-0.8B",
            "text_layers_kept": 12,
            "text_layers_total": 24,
            "hidden_size": 1024,
            "vision_layers": 12,
            "vision_hidden": 768,
            "num_classes": 356,
            "stripped": ["embed_tokens", "lm_head"],
        },
    }, int8_path)
    int8_size = int8_path.stat().st_size / 1024**2
    print(f"INT8 saved: {int8_path} ({int8_size:.0f}MB)")

    # Summary
    print(f"\n=== Export Summary ===")
    print(f"Original:  858M params, ~1637MB FP16")
    print(f"Stripped:  351M params (removed embed_tokens + lm_head)")
    print(f"FP16:      {fp16_size:.0f}MB")
    print(f"INT8:      {int8_size:.0f}MB")
    print(f"Budget:    400MB")
    print(f"Headroom:  {400 - int8_size:.0f}MB")
    print(f"Fits:      {'YES' if int8_size < 400 else 'NO'}")
    print(f"Accuracy:  {accuracy}")


if __name__ == "__main__":
    export()
