#!/usr/bin/env python3
"""Layer-by-layer parity check for MarkusNet language model.

Compares HF Qwen3.5 language_model against pure-PyTorch submission implementation
using the SAME checkpoint and SAME multimodal inputs_embeds/position_ids.
"""

import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float()
    return float(F.cosine_similarity(a, b).item())


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def load_run_fast_module(path: Path):
    spec = importlib.util.spec_from_file_location("run_fast_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_single_crop(image_path: Path, size: int = 224) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    cx, cy = w // 2, h // 2
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return img.crop((x1, y1, x2, y2))


def run_our_layers(mod, model, inputs_embeds, position_ids):
    hidden = inputs_embeds
    _, seqlen, _ = hidden.shape
    cos, sin = model.language._compute_rope(hidden, position_ids)
    causal_mask = torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), device=hidden.device, dtype=hidden.dtype),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)

    states = [hidden]
    for layer in model.language.layers:
        hidden = layer(hidden, cos, sin, causal_mask, attention_mask=None)
        states.append(hidden)

    final_hidden = mod.rms_norm(hidden, model.language.norm_weight, mod.TXT_RMS_EPS)
    return states, final_hidden


def run_hf_layers(hf_model, inputs_embeds, position_ids):
    lm = hf_model.model.language_model
    attn_mask = torch.ones(
        (inputs_embeds.shape[0], inputs_embeds.shape[1]),
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    out = lm(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = list(out.hidden_states) if out.hidden_states is not None else []
    return hidden_states, out.last_hidden_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-fast",
        default="/home/me/ht/nmiai/tasks/object-detection/submission-markusnet/run_fast.py",
    )
    ap.add_argument(
        "--pruned-dir",
        default="/home/me/ht/nmiai/tasks/object-detection/vlm-approach/pruned",
    )
    ap.add_argument(
        "--best-pt",
        default="/home/me/ht/nmiai/tasks/object-detection/vlm-approach/training_output/best/best.pt",
    )
    ap.add_argument(
        "--image",
        default="/home/me/ht/nmiai/tasks/object-detection/data-creation/data/clean_split/val/images/img_00002.jpg",
    )
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()

    device = torch.device("cpu")
    mod = load_run_fast_module(Path(args.run_fast))

    ckpt = torch.load(args.best_pt, map_location="cpu", weights_only=False)
    ms = ckpt["model_state"]

    # Build pure-PyTorch model
    our = mod.MarkusNet()
    our._load_vision(ms)
    our._load_language(ms)
    our.eval().float().to(device)

    # Build HF model and load same weights
    hf = AutoModelForImageTextToText.from_pretrained(
        args.pruned_dir,
        dtype=torch.float32,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    hf.load_state_dict(ms, strict=False)
    hf.eval().float().to(device)

    # Create one crop -> same multimodal inputs_embeds for both models
    crop = build_single_crop(Path(args.image), size=224)
    mean_t = torch.tensor(mod.QWEN_MEAN, device=device).view(3, 1, 1)
    std_t = torch.tensor(mod.QWEN_STD, device=device).view(3, 1, 1)
    pv, grid_thw = our._preprocess_single_crop(crop, device, mean_t, std_t)
    vis_embeds = our.vision(pv, [grid_thw])  # [1, num_img_tokens, hidden]

    h_patches = grid_thw[1]
    w_patches = grid_thw[2]
    llm_grid_h = h_patches // mod.VIS_SPATIAL_MERGE
    llm_grid_w = w_patches // mod.VIS_SPATIAL_MERGE
    num_img_tokens = vis_embeds.shape[1]

    token_ids = mod.CHAT_PREFIX_IDS + [mod.IMAGE_TOKEN_ID] * num_img_tokens + mod.CHAT_SUFFIX_IDS
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    inputs_embeds = our.language.embed_tokens(input_ids)
    image_mask = (input_ids == mod.IMAGE_TOKEN_ID).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, vis_embeds.to(inputs_embeds.dtype).reshape(-1))

    position_ids = our._build_position_ids(
        len(mod.CHAT_PREFIX_IDS),
        num_img_tokens,
        len(mod.CHAT_SUFFIX_IDS),
        1,
        llm_grid_h,
        llm_grid_w,
        device,
    )

    our_states, our_final = run_our_layers(mod, our, inputs_embeds, position_ids)
    hf_states, hf_final = run_hf_layers(hf, inputs_embeds, position_ids)

    print(f"num_layers_expected={mod.TXT_NUM_LAYERS}")
    print(f"our_states={len(our_states)} hf_states={len(hf_states)}")

    n = min(len(our_states), len(hf_states))
    first_bad = None
    for i in range(n):
        c = cosine(our_states[i], hf_states[i])
        m = max_abs(our_states[i], hf_states[i])
        print(f"layer_out_{i:02d}_cos={c:.10f} max_abs={m:.6e}")
        if first_bad is None and c < args.threshold:
            first_bad = i

    final_cos = cosine(our_final, hf_final)
    final_max_abs = max_abs(our_final, hf_final)
    print(f"final_cos={final_cos:.10f} final_max_abs={final_max_abs:.6e}")
    print("first_layer_below_threshold=", first_bad)


if __name__ == "__main__":
    main()
