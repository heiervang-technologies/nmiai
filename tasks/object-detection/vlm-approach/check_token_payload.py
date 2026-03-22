"""Validate preserved chat-template token embeddings in stripped MarkusNet checkpoints.

Usage:
  python check_token_payload.py \
    --base-ckpt training_output/best/best.pt \
    --ckpt exported/markusnet_351m_nf4.pt \
    --ckpt exported/markusnet_multitask_nf4.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

EXPECTED_TOKEN_IDS = [248045, 846, 198, 248053, 248054, 91037, 248046]
DEFAULT_CKPTS = [
    "exported/markusnet_351m_nf4.pt",
    "exported/markusnet_multitask_nf4.pt",
    "exported/markusnet_351m_fp16.pt",
    "exported/markusnet_351m_int8.pt",
]


def resolve_default_base_ckpt(root: Path, ckpt_path: Path) -> Path:
    name = ckpt_path.name.lower()
    if "multitask" in name:
        return root / "training_output_multitask" / "best" / "best.pt"
    return root / "training_output" / "best" / "best.pt"


def load_base_embed(base_ckpt: Path) -> torch.Tensor:
    base = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    try:
        return base["model_state"]["model.language_model.embed_tokens.weight"].float()
    except Exception as exc:  # pragma: no cover
        raise KeyError(
            f"Could not read base embed table from {base_ckpt}: {exc}"
        ) from exc


def validate_checkpoint(path: Path, base_embed: torch.Tensor, tol: float) -> tuple[bool, str]:
    ck = torch.load(path, map_location="cpu", weights_only=False)

    token_ids = ck.get("token_ids")
    token_embeds = ck.get("token_embeds")
    if token_ids is None or token_embeds is None:
        return False, "missing token_ids/token_embeds"

    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    token_embeds = token_embeds.float()

    if token_ids.tolist() != EXPECTED_TOKEN_IDS:
        return False, f"unexpected token_ids: {token_ids.tolist()}"

    if token_embeds.ndim != 2 or token_embeds.shape != (len(EXPECTED_TOKEN_IDS), base_embed.shape[1]):
        return False, f"bad token_embeds shape: {tuple(token_embeds.shape)}"

    ref = base_embed[token_ids]
    diff = (token_embeds - ref).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    if max_abs > tol:
        return False, f"payload mismatch max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}"

    return True, f"ok max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ckpt",
        type=Path,
        default=None,
        help="Optional base checkpoint override used for all exported checkpoints.",
    )
    parser.add_argument("--ckpt", action="append", type=Path, default=[])
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    root = Path(__file__).parent
    ckpts = args.ckpt or [root / rel for rel in DEFAULT_CKPTS]

    base_cache: dict[Path, torch.Tensor] = {}
    any_failed = False

    for ck in ckpts:
        ck_path = ck if ck.is_absolute() else (root / ck)
        if not ck_path.exists():
            print(f"FAIL {ck_path}: missing file")
            any_failed = True
            continue

        base_ckpt = args.base_ckpt if args.base_ckpt is not None else resolve_default_base_ckpt(root, ck_path)
        if base_ckpt not in base_cache:
            base_cache[base_ckpt] = load_base_embed(base_ckpt)
            print(f"Using base checkpoint: {base_ckpt}")

        ok, msg = validate_checkpoint(ck_path, base_cache[base_ckpt], args.tol)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {ck_path}: {msg}")
        any_failed = any_failed or (not ok)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
