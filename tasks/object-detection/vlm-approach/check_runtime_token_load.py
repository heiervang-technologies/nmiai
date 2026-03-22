"""Validate runtime token embedding fallback in submission-markusnet/run.py.

This ensures MarkusNet.load_checkpoint() correctly populates embed token rows
from token_ids/token_embeds when full embed_tokens are stripped.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

EXPECTED_TOKEN_IDS = [248045, 846, 198, 248053, 248054, 91037, 248046]
DEFAULT_CKPTS = [
    "exported/markusnet_351m_nf4.pt",
    "exported/markusnet_multitask_nf4.pt",
]


def resolve_base_ckpt(root: Path, export_ckpt: Path) -> Path:
    name = export_ckpt.name.lower()
    if "multitask" in name:
        return root / "training_output_multitask" / "best" / "best.pt"
    return root / "training_output" / "best" / "best.pt"


def load_markusnet_class(run_py: Path):
    spec = importlib.util.spec_from_file_location("markus_run", str(run_py))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MarkusNet


def validate_runtime_load(run_py: Path, export_ckpt: Path, tol: float) -> tuple[bool, str]:
    root = Path(__file__).parent
    base_ckpt = resolve_base_ckpt(root, export_ckpt)

    ck = torch.load(export_ckpt, map_location="cpu", weights_only=False)
    token_ids = ck.get("token_ids")
    token_embeds = ck.get("token_embeds")
    if token_ids is None or token_embeds is None:
        return False, "missing token_ids/token_embeds"

    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    if token_ids.tolist() != EXPECTED_TOKEN_IDS:
        return False, f"unexpected token_ids: {token_ids.tolist()}"

    MarkusNet = load_markusnet_class(run_py)
    model = MarkusNet()
    model.load_checkpoint(str(export_ckpt), torch.device("cpu"))

    loaded_rows = model.language.embed_tokens.weight.data[token_ids].float().cpu()
    payload_rows = token_embeds.float().cpu()

    payload_diff = (loaded_rows - payload_rows).abs()
    payload_max = float(payload_diff.max().item())

    base = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    base_embed = base["model_state"]["model.language_model.embed_tokens.weight"].float()
    base_rows = base_embed[token_ids]
    base_diff = (loaded_rows - base_rows).abs()
    base_max = float(base_diff.max().item())

    ok = payload_max <= tol and base_max <= tol
    msg = f"payload_max={payload_max:.3e} base_max={base_max:.3e}"
    return ok, msg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-py",
        type=Path,
        default=Path(__file__).parent.parent / "submission-markusnet" / "run.py",
    )
    parser.add_argument("--ckpt", action="append", type=Path, default=[])
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    root = Path(__file__).parent
    ckpts = args.ckpt or [root / rel for rel in DEFAULT_CKPTS]

    failed = False
    for ck in ckpts:
        ck_path = ck if ck.is_absolute() else (root / ck)
        if not ck_path.exists():
            print(f"FAIL {ck_path}: missing file")
            failed = True
            continue

        ok, msg = validate_runtime_load(args.run_py, ck_path, args.tol)
        print(f"{'PASS' if ok else 'FAIL'} {ck_path}: {msg}")
        failed = failed or (not ok)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
