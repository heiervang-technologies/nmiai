"""
Export MarkusNet-351M to NF4 (4-bit NormalFloat) quantization.

NF4 quantization:
- Groups of 64 elements share one FP16 scale factor
- Each weight mapped to nearest value in NF4 lookup table (16 values)
- Packed 2 weights per byte (4 bits each)
- Result: ~88MB from 335MB INT8 (or 670MB FP16)

The NF4 lookup table is designed for normally-distributed weights,
which transformer weights typically are.

Usage: uv run python export_nf4.py
"""

import functools
from pathlib import Path

import torch
import numpy as np

print = functools.partial(print, flush=True)

CHECKPOINT = Path(__file__).parent / "training_output" / "best" / "best.pt"
OUTPUT_DIR = Path(__file__).parent / "exported"

# NF4 lookup table (16 values, optimized for normal distribution)
# From QLoRA paper / bitsandbytes
NF4_TABLE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=torch.float32)

GROUP_SIZE = 64  # Elements per quantization group

SPECIAL_TOKEN_IDS = [248045, 846, 198, 248053, 248054, 91037, 248046]


def extract_token_payload(model_state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve only chat-template token embeddings when full embed table is stripped."""
    embed_key = "model.language_model.embed_tokens.weight"
    if embed_key not in model_state:
        raise KeyError(
            f"Missing {embed_key} in checkpoint; cannot export stripped NF4 safely."
        )

    embed_weight = model_state[embed_key]
    token_ids = torch.tensor(SPECIAL_TOKEN_IDS, dtype=torch.long)
    token_embeds = embed_weight[token_ids].to(torch.float16).cpu()
    return token_ids, token_embeds


def quantize_tensor_nf4(tensor: torch.Tensor) -> dict:
    """Quantize a 2D+ tensor to NF4 format.

    Returns dict with:
        'packed': uint8 tensor (2 values per byte)
        'scales': float16 tensor (one per group)
        'shape': original shape
    """
    original_shape = tensor.shape
    flat = tensor.float().reshape(-1)

    # Pad to multiple of GROUP_SIZE
    n = flat.numel()
    pad_n = (GROUP_SIZE - n % GROUP_SIZE) % GROUP_SIZE
    if pad_n > 0:
        flat = torch.cat([flat, torch.zeros(pad_n)])

    # Reshape into groups
    groups = flat.reshape(-1, GROUP_SIZE)
    num_groups = groups.shape[0]

    # Compute per-group absmax scale
    absmax = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = absmax.squeeze(1).to(torch.float16)

    # Normalize to [-1, 1]
    normalized = groups / absmax

    # Map each value to nearest NF4 index
    # Broadcast: [num_groups, GROUP_SIZE, 1] vs [1, 1, 16]
    distances = (normalized.unsqueeze(-1) - NF4_TABLE.unsqueeze(0).unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1)  # [num_groups, GROUP_SIZE] values in 0-15

    # Pack two 4-bit indices into one uint8
    indices_flat = indices.reshape(-1)
    assert indices_flat.numel() % 2 == 0
    high = indices_flat[0::2].to(torch.uint8)
    low = indices_flat[1::2].to(torch.uint8)
    packed = (high << 4) | low

    return {
        'packed': packed,
        'scales': scales,
        'shape': original_shape,
        'numel': n,  # Original element count (before padding)
    }


def dequantize_nf4(packed: torch.Tensor, scales: torch.Tensor,
                   shape: tuple, numel: int) -> torch.Tensor:
    """Dequantize NF4 packed tensor back to float16.

    This is the function that runs in the sandbox at inference time.
    Pure PyTorch, no external dependencies.
    """
    # Unpack 4-bit values
    high = (packed >> 4).to(torch.int64)
    low = (packed & 0x0F).to(torch.int64)
    indices = torch.stack([high, low], dim=1).reshape(-1)  # interleave

    # Lookup NF4 values
    nf4_table = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    ], dtype=torch.float32, device=packed.device)

    values = nf4_table[indices]  # [total_padded_elements]

    # Reshape into groups and multiply by scales
    values = values.reshape(-1, 64)  # GROUP_SIZE = 64
    values = values * scales.float().unsqueeze(1)

    # Flatten, truncate padding, reshape
    values = values.reshape(-1)[:numel]
    return values.reshape(shape).to(torch.float16)


def export():
    print("=== MarkusNet NF4 Export ===")

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state"]
    cls_state = ckpt["cls_head_state"]
    accuracy = ckpt.get("accuracy", 0)
    print(f"Accuracy: {accuracy:.4f}")

    token_ids, token_embeds = extract_token_payload(model_state)
    print(f"Preserving {token_ids.numel()} special token embeddings for template tokens")

    # Strip embed_tokens and lm_head
    nf4_state = {}
    kept_fp16 = {}  # Small tensors stay FP16
    total_original = 0
    total_nf4 = 0

    for k, v in model_state.items():
        if "embed_tokens" in k or "lm_head" in k:
            print(f"  SKIP: {k} ({v.numel()/1e6:.1f}M)")
            continue

        total_original += v.numel() * 2  # FP16 bytes

        if v.dim() >= 2 and v.numel() >= GROUP_SIZE:
            # Quantize to NF4
            q = quantize_tensor_nf4(v)
            nf4_state[k] = q
            packed_bytes = q['packed'].numel()
            scale_bytes = q['scales'].numel() * 2  # FP16
            total_nf4 += packed_bytes + scale_bytes
        else:
            # Keep small tensors as FP16 (norms, biases)
            kept_fp16[k] = v.to(torch.float16)
            total_nf4 += v.numel() * 2

    # Cls head stays FP16 (tiny)
    cls_fp16 = {k: v.to(torch.float16) for k, v in cls_state.items()}
    for v in cls_fp16.values():
        total_nf4 += v.numel() * 2

    print(f"\nOriginal (FP16, stripped): {total_original/1024**2:.0f} MB")
    print(f"NF4 packed estimate: {total_nf4/1024**2:.0f} MB")
    print(f"Compression ratio: {total_original/total_nf4:.1f}x")

    # Verify dequantization roundtrip on one tensor
    print("\nVerifying dequant roundtrip...")
    test_key = [k for k in nf4_state.keys() if 'weight' in k][0]
    test_q = nf4_state[test_key]
    original = model_state[test_key].float()
    reconstructed = dequantize_nf4(
        test_q['packed'], test_q['scales'], test_q['shape'], test_q['numel']
    ).float()
    rmse = (original - reconstructed).pow(2).mean().sqrt().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        original.reshape(1, -1), reconstructed.reshape(1, -1)
    ).item()
    print(f"  Test tensor: {test_key}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "markusnet_351m_nf4.pt"

    torch.save({
        'nf4_state': nf4_state,
        'fp16_state': kept_fp16,
        'cls_head_state': cls_fp16,
        'accuracy': accuracy,
        'global_step': ckpt.get('global_step', 0),
        'quantization': 'nf4',
        'group_size': GROUP_SIZE,
        'token_ids': token_ids,
        'token_embeds': token_embeds,
        'architecture': {
            'base': 'Qwen3.5-0.8B',
            'text_layers_kept': 12,
            'text_layers_total': 24,
            'hidden_size': 1024,
            'vision_layers': 12,
            'vision_hidden': 768,
            'num_classes': 356,
            'stripped': ['embed_tokens', 'lm_head'],
            'preserved_token_ids': SPECIAL_TOKEN_IDS,
        },
    }, output_path)

    file_size = output_path.stat().st_size / 1024**2
    print(f"\nSaved: {output_path}")
    print(f"File size: {file_size:.1f} MB")
    print(f"+ YOLO ONNX (~132 MB) = {file_size + 132:.1f} MB total")
    print(f"Fits 400 MB: {'YES' if file_size + 132 < 400 else 'NO'} (headroom: {400 - file_size - 132:.0f} MB)")

    # Also generate the dequant snippet for run.py
    print(f"\n=== DEQUANT CODE FOR run.py ===")
    print("""
def load_nf4_checkpoint(path, device='cuda'):
    NF4_TABLE = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    ], dtype=torch.float32, device=device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = {}

    # Dequantize NF4 tensors
    for k, q in ckpt['nf4_state'].items():
        packed = q['packed'].to(device)
        scales = q['scales'].to(device)
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=1).reshape(-1)
        values = NF4_TABLE[indices].reshape(-1, 64) * scales.float().unsqueeze(1)
        state[k] = values.reshape(-1)[:q['numel']].reshape(q['shape']).half()

    # Add FP16 tensors
    for k, v in ckpt['fp16_state'].items():
        state[k] = v.to(device)

    return state, ckpt['cls_head_state']
""")


if __name__ == "__main__":
    export()
