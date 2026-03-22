"""Pre-compute SigLIP text embeddings for 356 grocery categories.

Uses transformers (offline only) to encode category names.
Output: text_embeddings.pth with shape [356, embed_dim] + metadata.
"""
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

SCRIPT_DIR = Path(__file__).parent
ANNO_PATH = SCRIPT_DIR / "data-creation/data/coco_dataset/train/annotations.json"
OUTPUT_PATH = SCRIPT_DIR / "text_embeddings.pth"

# The SO400M model - strongest SigLIP variant
MODEL_ID = "google/siglip-so400m-patch14-384"


def load_categories():
    with open(ANNO_PATH) as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda c: c["id"])
    assert len(cats) == 356
    assert cats[0]["id"] == 0 and cats[-1]["id"] == 355
    return cats


def build_prompts(cats):
    """Build text prompts for each category. Try multiple prompt templates."""
    prompts = []
    for c in cats:
        name = c["name"]
        # SigLIP works well with simple descriptions
        prompts.append(f"a photo of {name}")
    return prompts


def build_prompts_multi(cats):
    """Multiple prompts per category for ensemble."""
    templates = [
        "a photo of {name}",
        "a photo of {name} on a store shelf",
        "{name} grocery product",
        "a product package of {name}",
    ]
    all_prompts = []
    for c in cats:
        name = c["name"]
        cat_prompts = [t.format(name=name) for t in templates]
        all_prompts.append(cat_prompts)
    return all_prompts


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cats = load_categories()
    print(f"Loaded {len(cats)} categories")

    # Load SigLIP text encoder
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    text_model = model.text_model
    embed_dim = model.config.text_config.hidden_size
    print(f"Text embedding dim: {embed_dim}")

    # Single prompt embeddings
    prompts = build_prompts(cats)
    print(f"Encoding {len(prompts)} single prompts...")

    # Batch encode
    batch_size = 64
    all_embeds = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        # SigLIP text model output
        outputs = text_model(**inputs)
        # Pool: use the EOS token embedding (last non-padding token)
        # For SigLIP, pooler_output is the right thing
        embeds = outputs.pooler_output
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu())
        print(f"  Batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")

    single_embeds = torch.cat(all_embeds, dim=0)  # [356, embed_dim]
    print(f"Single embeddings shape: {single_embeds.shape}")

    # Multi-prompt ensemble embeddings
    multi_prompts = build_prompts_multi(cats)
    print(f"Encoding multi-prompt ensemble ({len(multi_prompts[0])} templates)...")

    ensemble_embeds = []
    for cat_idx, cat_prompts in enumerate(multi_prompts):
        inputs = tokenizer(cat_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = text_model(**inputs)
        embeds = outputs.pooler_output
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        # Average the normalized embeddings, then re-normalize
        mean_embed = embeds.mean(dim=0)
        mean_embed = mean_embed / mean_embed.norm()
        ensemble_embeds.append(mean_embed.cpu())

    ensemble_embeds = torch.stack(ensemble_embeds, dim=0)  # [356, embed_dim]
    print(f"Ensemble embeddings shape: {ensemble_embeds.shape}")

    # Save
    cat_names = [c["name"] for c in cats]
    cat_ids = [c["id"] for c in cats]

    save_dict = {
        "single_embeddings": single_embeds,       # [356, embed_dim]
        "ensemble_embeddings": ensemble_embeds,    # [356, embed_dim]
        "category_names": cat_names,
        "category_ids": cat_ids,
        "embed_dim": embed_dim,
        "model_id": MODEL_ID,
        "num_classes": 356,
    }
    torch.save(save_dict, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")

    # Also save just the embedding matrix for submission
    torch.save(ensemble_embeds, SCRIPT_DIR / "text_embeddings_matrix.pth")
    print(f"Saved matrix to text_embeddings_matrix.pth")

    # Quick sanity: compute similarity between first few categories
    sim = ensemble_embeds[:5] @ ensemble_embeds[:5].T
    print("\nSimilarity matrix (first 5 categories):")
    for i in range(5):
        row = " ".join(f"{sim[i, j]:.3f}" for j in range(5))
        print(f"  [{i}] {cat_names[i][:40]:40s} | {row}")


if __name__ == "__main__":
    main()
