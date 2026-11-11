"""
Prepare gt-harmbench-cleaned.csv (the 1,535-row release) for HuggingFace upload.

Transformations (auditable):
  1. Remove Matching Pennies rows (no pure Nash equilibrium; a no-op on the
     cleaned release, kept as a guard).
  2. Drop internal review/annotation columns.
  3. Drop redundant numeric target columns (computable from payoffs).
  4. Rename columns to snake_case.
  5. Save cleaned CSV to data/gt-harmbench-hf.csv.
"""

from pathlib import Path

import pandas as pd

INPUT = Path("data/gt-harmbench-cleaned.csv")
OUTPUT = Path("data/gt-harmbench-hf.csv")

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
df = pd.read_csv(INPUT)
print(f"Loaded:  {len(df):,} rows, {len(df.columns)} columns")

# ------------------------------------------------------------------
# 1. Remove Matching Pennies (no pure Nash)
# ------------------------------------------------------------------
mask_mp = df["formal_game"] == "Matching pennies"
print(f"  Dropping {mask_mp.sum()} Matching Pennies rows")
df = df[~mask_mp].reset_index(drop=True)

# ------------------------------------------------------------------
# 2. Drop internal review / annotation columns
# ------------------------------------------------------------------
DROP_REVIEW = [
    "quality_score",
    "quality_justification",
    "equilibria_score",
    "equilibria_justification",
    "issues",
    "overall_comment",
    "justification",
]
# ------------------------------------------------------------------
# 3. Drop redundant numeric target columns
# ------------------------------------------------------------------
DROP_NUMERIC = [
    "nash_social_welfare",
    "max_utilitarian",
    "max_rawlsian",
]

to_drop = DROP_REVIEW + DROP_NUMERIC
missing = [c for c in to_drop if c not in df.columns]
if missing:
    raise ValueError(f"Columns not found: {missing}")

print(f"  Dropping columns: {to_drop}")
df = df.drop(columns=to_drop)

# ------------------------------------------------------------------
# 4. Rename to snake_case
# ------------------------------------------------------------------
RENAME = {
    "Ev_ID": "ev_id",
    "Domain": "domain",
    "Risk category": "risk_category",
    "Risk subcategory": "risk_subcategory",
    "Description": "description",
}
df = df.rename(columns=RENAME)

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print(f"Output:  {len(df):,} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")
print()
print("Null counts:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print()
print("formal_game distribution:")
print(df["formal_game"].value_counts().to_string())

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
df.to_csv(OUTPUT, index=False)
print(f"\nSaved to {OUTPUT}")
