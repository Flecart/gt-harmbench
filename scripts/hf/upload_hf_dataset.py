"""
Upload gt-harmbench to HuggingFace as dataset Jinesis/gt-harmbench.

Steps:
  1. Load cleaned CSV into a HuggingFace Dataset.
  2. Push data to hub (creates repo if needed).
  3. Upload dataset card (data/hf_dataset_card.md) as README.md.

Usage:
    uv run python scripts/upload_hf_dataset.py [--public]

Requires: huggingface_hub, datasets
    uv run pip install huggingface_hub datasets
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

REPO_ID = "Jinesis/gt-harmbench"
DATA_CSV = Path("data/gt-harmbench-hf.csv")
CARD_MD = Path("data/hf_dataset_card.md")

parser = argparse.ArgumentParser()
parser.add_argument("--public", action="store_true", help="Make the repo public (default: private)")
args = parser.parse_args()

private = not args.public

# ------------------------------------------------------------------
# 1. Push data
# ------------------------------------------------------------------
print(f"Loading {DATA_CSV} …")
df = pd.read_csv(DATA_CSV)
print(f"  {len(df):,} rows, {len(df.columns)} columns")

ds = Dataset.from_pandas(df, preserve_index=False)

print(f"Pushing data to {REPO_ID} (private={private}) …")
ds.push_to_hub(REPO_ID, private=private)
print("  Data pushed.")

# ------------------------------------------------------------------
# 2. Upload dataset card
# ------------------------------------------------------------------
print(f"Uploading dataset card from {CARD_MD} …")
api = HfApi()
api.upload_file(
    path_or_fileobj=str(CARD_MD),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("  Dataset card uploaded.")

print(f"\nDone. View at: https://huggingface.co/datasets/{REPO_ID}")
print("To make public later:")
print("  python scripts/upload_hf_dataset.py --public")
