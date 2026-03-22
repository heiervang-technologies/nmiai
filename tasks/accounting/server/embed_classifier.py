"""
Embedding-based task family classifier using Qwen3-Embedding.
Replaces/supplements keyword-based planner for ambiguous prompts.

Pre-computes reference embeddings for each family from real prompts,
then classifies new prompts by cosine similarity.
"""

import json
import logging
import os
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

_model = None
_ref_embeddings = None  # {family: np.array of shape (n_refs, dim)}
_ref_labels = None

CACHE_FILE = Path(__file__).parent / "embed_refs.npz"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        cache = os.path.expanduser("~/Models/hf_cache")
        os.makedirs(cache, exist_ok=True)
        model_name = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
        _model = SentenceTransformer(
            model_name,
            device="cuda",
            trust_remote_code=True,
            cache_folder=cache,
        )
        log.info(f"Loaded {model_name} on CUDA")
    return _model


def build_reference_embeddings(log_dir: str = "/tmp/accounting-logs"):
    """Build reference embeddings from real competition logs."""
    log_path = Path(log_dir)
    summary = log_path / "summary.jsonl"
    if not summary.exists():
        log.error("No summary.jsonl found")
        return

    # Collect prompts by family
    family_prompts = {}
    for line in summary.read_text().strip().split("\n"):
        s = json.loads(line)
        ts = s["ts"]
        family = s.get("family")
        if not family:
            continue
        detail = log_path / f"{ts}.json"
        if not detail.exists():
            continue
        d = json.loads(detail.read_text())
        prompt = d["prompt"]
        if len(prompt) < 10:  # Skip garbage
            continue
        family_prompts.setdefault(family, []).append(prompt)

    model = _get_model()
    ref_data = {}
    for family, prompts in family_prompts.items():
        embeddings = model.encode(prompts, show_progress_bar=False, batch_size=32)
        ref_data[family] = embeddings
        log.info(f"  {family}: {len(prompts)} refs, shape {embeddings.shape}")

    # Save to cache
    np.savez(CACHE_FILE, **{f"family_{k}": v for k, v in ref_data.items()})
    log.info(f"Saved reference embeddings to {CACHE_FILE}")
    return ref_data


def _load_refs():
    global _ref_embeddings
    if _ref_embeddings is not None:
        return _ref_embeddings

    if CACHE_FILE.exists():
        data = np.load(CACHE_FILE)
        _ref_embeddings = {}
        for key in data.files:
            family = key.replace("family_", "")
            _ref_embeddings[family] = data[key]
        log.info(f"Loaded cached refs: {list(_ref_embeddings.keys())}")
        return _ref_embeddings

    log.info("Building reference embeddings from logs...")
    _ref_embeddings = build_reference_embeddings()
    return _ref_embeddings


def classify(prompt: str, top_k: int = 3) -> list[dict]:
    """
    Classify a prompt by cosine similarity to reference embeddings.

    Returns list of {family, score, confidence} sorted by score desc.
    """
    refs = _load_refs()
    if not refs:
        return []

    model = _get_model()
    query_emb = model.encode([prompt], show_progress_bar=False)[0]
    query_norm = query_emb / np.linalg.norm(query_emb)

    scores = []
    for family, ref_embs in refs.items():
        # Cosine similarity to each reference, take max
        ref_norms = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)
        sims = ref_norms @ query_norm
        max_sim = float(np.max(sims))
        mean_sim = float(np.mean(np.sort(sims)[-3:]))  # Top-3 mean
        scores.append({"family": family, "score": mean_sim, "max_score": max_sim})

    scores.sort(key=lambda x: -x["score"])
    # Add confidence based on gap between top-1 and top-2
    if len(scores) >= 2:
        gap = scores[0]["score"] - scores[1]["score"]
        scores[0]["confidence"] = "high" if gap > 0.05 else "medium" if gap > 0.02 else "low"
    elif scores:
        scores[0]["confidence"] = "medium"

    return scores[:top_k]


if __name__ == "__main__":
    # Build refs and test
    import sys
    logging.basicConfig(level=logging.INFO)

    if "--build" in sys.argv:
        build_reference_embeddings()
        print("Done building references")
    else:
        # Test classification
        test_prompts = [
            "Registrer timer på prosjektet",  # Should be timesheet, not project
            "Opprett dimensjon Produktlinje",  # Should be voucher, not product
            "Crie uma fatura para o cliente Solmar Lda",  # invoice or customer?
            "Gjer månavslutninga for mars 2026",  # annual_close or cost_analysis?
            "Enregistrez une note de frais de déplacement",  # travel_expense
        ]
        for p in test_prompts:
            results = classify(p)
            top = results[0] if results else {"family": "?", "score": 0}
            print(f"  {top['family']:20s} ({top['score']:.3f}) | {p[:60]}")
