"""Inter-rater agreement for the two human validation studies.

Study 1 — game-type classification: two annotators independently label the
formal game of 30 scenarios (5 per game, seed 42) from 7 options.

Study 2 — binary quality rating: two annotators label 100 scenarios
(50 drawn from the 'good' pool, 50 from the 'bad' pool) as good or bad.

Both report Cohen's kappa, where expected agreement is derived from each
annotator's own marginal distribution.
"""

import pandas as pd

VALIDATION_DIR = "artifacts/validation"


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two equal-length series of categorical labels."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    observed = (a == b).mean()
    # Expected agreement from the raters' marginals, not a hardcoded constant:
    # sum over categories of P(rater A picks c) * P(rater B picks c).
    labels = set(a) | set(b)
    expected = sum((a == c).mean() * (b == c).mean() for c in labels)
    return (observed - expected) / (1 - expected)


def report(name: str, a: pd.Series, b: pd.Series) -> None:
    agree = int((a.reset_index(drop=True) == b.reset_index(drop=True)).sum())
    print(f"{name}")
    print(f"  n               = {len(a)}")
    print(f"  raw agreement   = {agree}/{len(a)} = {100 * agree / len(a):.1f}%")
    print(f"  Cohen's kappa   = {cohen_kappa(a, b):.4f}\n")


def main() -> None:
    # Study 1 — game-type classification.
    g1 = pd.read_csv(f"{VALIDATION_DIR}/inferences_gt-harmbench.csv")
    g2 = pd.read_csv(f"{VALIDATION_DIR}/inferences_gt-harmbench_2.csv")
    merged = g1[["id", "selected_game"]].merge(
        g2[["id", "selected_game"]], on="id", suffixes=("_1", "_2")
    )
    report("Study 1 — game-type classification",
           merged["selected_game_1"], merged["selected_game_2"])

    # Study 2 — binary good/bad quality rating.
    q1 = pd.read_csv(f"{VALIDATION_DIR}/ratings_merged.csv")
    q2 = pd.read_csv(f"{VALIDATION_DIR}/ratings_merged_2.csv")
    report("Study 2 — binary quality rating", q1["rating"], q2["rating"])

    # Not agreement: how often each annotator's rating matched the pool a
    # scenario was drawn from. Useful as a sanity check, but it is accuracy
    # against a provisional label, not inter-rater reliability.
    for i, q in enumerate((q1, q2), start=1):
        acc = (q["rating"] == q["source"]).mean()
        print(f"Annotator {i} vs. original good/bad pool: {100 * acc:.1f}% match")


if __name__ == "__main__":
    main()
