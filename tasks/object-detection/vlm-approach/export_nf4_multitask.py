"""Export the multitask best checkpoint to NF4."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from export_nf4 import quantize_tensor_nf4, dequantize_nf4, NF4_TABLE, GROUP_SIZE

import functools
from pathlib import Path
import torch

print = functools.partial(print, flush=True)

CHECKPOINT = Path(__file__).parent / "training_output_multitask" / "best" / "best.pt"
OUTPUT = Path(__file__).parent / "exported" / "markusnet_multitask_nf4.pt"

print("=== MarkusNet Multitask NF4 Export ===")
ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
model_state = ckpt["model_state"]
cls_state = ckpt["cls_head_state"]
val_acc = ckpt.get("val_acc", ckpt.get("accuracy", 0))
print(f"Val accuracy: {val_acc:.4f}, Step: {ckpt.get('global_step', '?')}")

nf4_state = {}
kept_fp16 = {}
total_original = 0
total_nf4 = 0

for k, v in model_state.items():
    if "embed_tokens" in k or "lm_head" in k:
        print(f"  SKIP: {k} ({v.numel()/1e6:.1f}M)")
        continue
    total_original += v.numel() * 2
    if v.dim() >= 2 and v.numel() >= GROUP_SIZE:
        q = quantize_tensor_nf4(v)
        nf4_state[k] = q
        total_nf4 += q["packed"].numel() + q["scales"].numel() * 2
    else:
        kept_fp16[k] = v.to(torch.float16)
        total_nf4 += v.numel() * 2

cls_fp16 = {k: v.to(torch.float16) for k, v in cls_state.items()}
for v in cls_fp16.values():
    total_nf4 += v.numel() * 2

print(f"Original (FP16, stripped): {total_original/1024**2:.0f} MB")
print(f"NF4 packed estimate: {total_nf4/1024**2:.0f} MB")
print(f"Compression ratio: {total_original/total_nf4:.1f}x")

# Verify
test_key = [k for k in nf4_state.keys() if "weight" in k][0]
test_q = nf4_state[test_key]
original = model_state[test_key].float()
reconstructed = dequantize_nf4(test_q["packed"], test_q["scales"], test_q["shape"], test_q["numel"]).float()
rmse = (original - reconstructed).pow(2).mean().sqrt().item()
cos_sim = torch.nn.functional.cosine_similarity(original.reshape(1, -1), reconstructed.reshape(1, -1)).item()
print(f"Verify: RMSE={rmse:.6f}, cos_sim={cos_sim:.6f}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "nf4_state": nf4_state,
    "fp16_state": kept_fp16,
    "cls_head_state": cls_fp16,
    "val_acc": val_acc,
    "global_step": ckpt.get("global_step", 0),
    "quantization": "nf4",
    "group_size": GROUP_SIZE,
    "architecture": {
        "base": "Qwen3.5-0.8B",
        "text_layers_kept": 12,
        "hidden_size": 1024,
        "vision_layers": 12,
        "num_classes": 356,
        "stripped": ["embed_tokens", "lm_head"],
    },
}, OUTPUT)

file_size = OUTPUT.stat().st_size / 1024**2
print(f"Saved: {OUTPUT}")
print(f"File size: {file_size:.1f} MB")
print(f"+ YOLO ONNX (~132 MB) = {file_size + 132:.1f} MB total")
print(f"Fits 420 MB: {'YES' if file_size + 132 < 420 else 'NO'}")
