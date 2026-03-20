"""
Prune Qwen3.5-0.8B for classification submission.

Keep: vision encoder (12 blocks) + merger + first K text transformer blocks + classification head
Drop: embed_tokens, lm_head, text blocks [K:]

This is the CORRECT model: Qwen/Qwen3.5-0.8B (native multimodal, NOT Qwen2.5-VL)
"""

import torch
from transformers import AutoModelForImageTextToText, AutoConfig
from pathlib import Path
import json


def inspect_model(model):
    """Print all modules with param counts."""
    print("=== Model Architecture ===")
    total = 0
    components = {}
    for name, param in model.named_parameters():
        size = param.numel()
        total += size
        # Group by top-level component
        top = name.split(".")[0] + "." + name.split(".")[1] if "." in name else name
        components[top] = components.get(top, 0) + size

    print(f"\nTop-level components:")
    for name, count in sorted(components.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count/1e6:.1f}M params = {count*2/1024**2:.1f} MB FP16")
    print(f"\nTotal: {total/1e6:.1f}M params = {total*2/1024**2:.1f} MB FP16")
    return total


def prune_and_save(model, keep_text_blocks=10, output_dir=Path(".")):
    """Extract pruned state dict: vision + merger + K text blocks + cls head."""

    pruned = {}
    kept_params = 0
    dropped_params = 0

    for name, param in model.named_parameters():
        keep = False

        # Keep all vision encoder params
        if "visual" in name:
            keep = True

        # Keep merger/projector
        elif "merger" in name or "projector" in name:
            keep = True

        # Keep first K text transformer blocks
        elif "model.layers." in name:
            # Extract layer index
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    if layer_idx < keep_text_blocks:
                        keep = True
                    break

        # Keep norms that aren't tied to dropped layers
        elif "model.norm" in name or "model.rotary" in name:
            keep = True

        # Drop: embed_tokens, lm_head, layers >= K
        # (embed_tokens and lm_head are explicitly not kept)

        if keep:
            pruned[name] = param.data.cpu()
            kept_params += param.numel()
        else:
            dropped_params += param.numel()

    print(f"\n=== Pruning Results (keep_text_blocks={keep_text_blocks}) ===")
    print(f"Kept: {kept_params/1e6:.1f}M params = {kept_params*2/1024**2:.1f} MB FP16")
    print(f"Dropped: {dropped_params/1e6:.1f}M params = {dropped_params*2/1024**2:.1f} MB FP16")

    # Save FP16
    fp16_path = output_dir / f"pruned_fp16_{keep_text_blocks}blocks.pth"
    torch.save(pruned, fp16_path)
    fp16_size = fp16_path.stat().st_size / 1024**2
    print(f"FP16 file: {fp16_size:.1f} MB")

    # Quantize to INT8 via converting to half then to int8 manually
    # For proper INT8: save float16 and measure
    # Real INT8 would be done via ONNX or torch quantization at inference
    int8_est = kept_params / 1024**2  # 1 byte per param
    print(f"INT8 estimate: {int8_est:.1f} MB")
    print(f"INT8 + YOLO (130MB): {int8_est + 130:.1f} MB")
    print(f"Fits 400MB budget: {'YES' if int8_est + 130 < 400 else 'NO'} (headroom: {400 - int8_est - 130:.0f} MB)")

    return pruned, kept_params


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading Qwen3.5-0.8B...")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    total = inspect_model(model)

    output_dir = Path("/home/me/ht/nmiai/tasks/object-detection/vlm-approach/pruned")
    output_dir.mkdir(exist_ok=True)

    # Test multiple pruning levels
    for k in [6, 8, 10, 12]:
        prune_and_save(model, keep_text_blocks=k, output_dir=output_dir)
        print()


if __name__ == "__main__":
    main()
