#!/usr/bin/env python3
"""Upload pretrain subset to HuggingFace as a dataset.

Usage: python upload_pretrain_hf.py
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

DATASET_DIR = Path(__file__).parent / "data-creation/data/pretrain_subset"
REPO_ID = "heiertech/nmiai-pretrain"


def main():
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
        print(f"Repo ready: {REPO_ID}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload the dataset
    print(f"Uploading {DATASET_DIR} to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(DATASET_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Pre-training subset: 2045 train (Polish+Grocery+SKU110K) + 248 val (competition)",
    )
    print(f"Done! https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
