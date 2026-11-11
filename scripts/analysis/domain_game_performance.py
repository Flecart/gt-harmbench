"""Analyse model performance per domain and game type from a single eval log.

Usage:
    uv run python -m scripts.analysis.domain_game_performance --log <path.eval>

The script joins sample IDs with GT-HarmBench + MIT AI Risk Database to map
each scenario to one of the 7 risk domains, then aggregates Nash / utilitarian
/ Rawlsian accuracy broken down by domain and by formal game type.

``Matching pennies`` rows are dropped before any output (aligned with
``domain_distribution.py``).
"""

import click
import pandas as pd
from inspect_ai.log import read_eval_log

GT_CSV = "data/gt-harmbench-cleaned.csv"
MIT_CSV = "data/generation/mit.csv"

METRICS = ["nash", "utilitarian", "rawlsian"]

# Games excluded from all tables (same convention as ``domain_distribution.py``).
EXCLUDE_GAMES = frozenset({"Matching pennies"})


def load_domain_map() -> pd.DataFrame:
    """Return a DataFrame (id -> Domain, formal_game) built from GT + MIT CSVs."""
    gt = pd.read_csv(GT_CSV, dtype={"id": str})
    mit = pd.read_csv(MIT_CSV)[["Ev_ID", "Domain"]].drop_duplicates()
    merged = gt.merge(mit, on="Ev_ID", how="left")
    return merged[["id", "Domain", "formal_game"]].copy()


def extract_samples(log_path: str) -> pd.DataFrame:
    """Read an eval log and return a DataFrame with one row per sample."""
    log = read_eval_log(log_path)
    scorer_name = next(iter(log.samples[0].scores))

    rows = []
    for sample in log.samples:
        score_val = sample.scores[scorer_name].value
        if not isinstance(score_val, dict):
            continue
        rows.append(
            {
                "id": str(sample.id),
                "nash": score_val.get("nash"),
                "utilitarian": score_val.get("utilitarian"),
                "rawlsian": score_val.get("rawlsian"),
                "formal_game_log": sample.metadata.get("formal_game"),
            }
        )

    return pd.DataFrame(rows)


def accuracy_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute mean accuracy for each metric grouped by *group_col*."""
    agg = (
        df.groupby(group_col)[METRICS]
        .agg(["mean", "count"])
        .round(3)
    )
    # Flatten MultiIndex columns
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    # Keep only means plus a single sample count (same for all metrics)
    result = agg[[f"{m}_mean" for m in METRICS]].copy()
    result.columns = [m.capitalize() + " Acc." for m in METRICS]
    result["N"] = agg["nash_count"]
    return result.sort_values("Nash Acc.", ascending=False)


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    print()


def pivot_domain_game_with_weighted_margins(
    df: pd.DataFrame,
    value_col: str,
    domain_col: str = "domain_key",
) -> pd.DataFrame:
    """Domain × game mean table with count-weighted average row and column.

    Row ``Weighted Avg.``: for each game column, mean over all samples with that game
    (same weight as weighting each domain cell by its sample count).

    Column ``Weighted Avg.``: for each domain row, mean over all samples in that domain.

    Bottom-right: overall mean for *value_col* on *df*.
    """
    pivot = df.pivot_table(
        values=value_col,
        index=domain_col,
        columns="game",
        aggfunc="mean",
    ).round(3)

    out = pivot.copy()
    domain_means = df.groupby(domain_col)[value_col].mean().round(3)
    out["Weighted Avg."] = domain_means.reindex(out.index)

    game_means = df.groupby("game")[value_col].mean().round(3)
    overall = round(float(df[value_col].mean()), 3)
    row_vals: dict[str, float] = {}
    for col in out.columns:
        if col == "Weighted Avg.":
            row_vals[col] = overall
        else:
            row_vals[col] = float(game_means[col]) if col in game_means.index else float("nan")
    out.loc["Weighted Avg."] = pd.Series(row_vals)

    return out.reindex(sort_domain_key_index(out.index))


def sort_domain_key_index(idx: pd.Index) -> list:
    """MIT domains 1–7 first, then ``(No domain)``, then ``Weighted Avg.``."""
    rows = [x for x in idx if x != "Weighted Avg."]

    def sort_key(r: object) -> tuple:
        if r == "(No domain)":
            return (1, 999)
        if isinstance(r, str) and len(r) > 0 and r[0].isdigit():
            part = r.split(".", 1)[0]
            return (0, int(part) if part.isdigit() else 999)
        return (1, 0)

    ordered = sorted(rows, key=sort_key)
    if "Weighted Avg." in idx:
        ordered.append("Weighted Avg.")
    return ordered


def short_domain_column_header(domain_label: str) -> str:
    """MIT domains -> leading number; keep ``(No domain)`` readable."""
    if domain_label == "(No domain)":
        return "No domain"
    if (
        isinstance(domain_label, str)
        and len(domain_label) > 0
        and domain_label[0].isdigit()
    ):
        return domain_label.split(".")[0]
    return domain_label


@click.command()
@click.option(
    "--log",
    "log_path",
    required=True,
    help="Path to the .eval log file to analyse.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Optional CSV path to save the combined results table.",
)
def main(log_path: str, output: str | None) -> None:
    """Print model performance by domain and game type for a single eval log."""

    samples_df = extract_samples(log_path)
    domain_map = load_domain_map()
    df = samples_df.merge(domain_map, on="id", how="left")

    # Prefer the formal_game from metadata (already in samples)
    df["game"] = df["formal_game_log"].fillna(df["formal_game"])
    df["domain_key"] = df["Domain"].fillna("(No domain)")

    df = df.dropna(subset=METRICS)
    n_matching_pennies = df["game"].isin(EXCLUDE_GAMES).sum()
    df = df[~df["game"].isin(EXCLUDE_GAMES)].copy()

    print(f"\nLoading log: {log_path}")
    print(f"  Samples loaded: {len(samples_df)}")
    print(f"  Excluded Matching pennies: {n_matching_pennies}")
    print(f"  Samples in analysis: {len(df)}")
    print(f"  Samples with domain info:  {df['Domain'].notna().sum()}")
    print(f"  Samples with no domain:    {df['Domain'].isna().sum()}")

    # ------------------------------------------------------------------ #
    # Performance by domain (+ rows without MIT domain)                   #
    # ------------------------------------------------------------------ #
    by_domain = accuracy_table(df.dropna(subset=["game"]), "domain_key")
    print_table("Performance by Risk Domain", by_domain)

    # ------------------------------------------------------------------ #
    # Performance by formal game type                                      #
    # ------------------------------------------------------------------ #
    by_game = accuracy_table(df.dropna(subset=["game"]), "game")
    print_table("Performance by Formal Game Type", by_game)

    # ------------------------------------------------------------------ #
    # Count table: game (rows) × domain (columns)                         #
    # ------------------------------------------------------------------ #
    perf_game = df.dropna(subset=["game"])
    count_pivot = perf_game.pivot_table(
        values="nash",
        index="game",
        columns="domain_key",
        aggfunc="count",
        fill_value=0,
    ).astype(int)
    count_pivot["Total"] = count_pivot.sum(axis=1)
    count_pivot.loc["Total"] = count_pivot.sum()
    count_pivot.columns = [
        short_domain_column_header(c) if c != "Total" else "Total"
        for c in count_pivot.columns
    ]
    _dom_cols = [c for c in count_pivot.columns if c not in ("Total", "No domain")]
    _dom_cols.sort(key=lambda x: int(x) if str(x).isdigit() else 999)
    count_pivot = count_pivot[
        _dom_cols
        + (["No domain"] if "No domain" in count_pivot.columns else [])
        + (["Total"] if "Total" in count_pivot.columns else [])
    ]
    print_table(
        "Sample Counts — Game × Domain  (1–7 = MIT risk areas; No domain = unmatched Ev_ID)",
        count_pivot,
    )

    # ------------------------------------------------------------------ #
    # Performance by domain × game (pivots + weighted margins)          #
    # ------------------------------------------------------------------ #
    for metric, title in (
        ("nash", "Nash Accuracy — Domain × Game (pivot)"),
        ("utilitarian", "Utilitarian Accuracy — Domain × Game (pivot)"),
        ("rawlsian", "Rawlsian Accuracy — Domain × Game (pivot)"),
    ):
        pivot_m = pivot_domain_game_with_weighted_margins(perf_game, metric)
        print_table(title, pivot_m)

    # ------------------------------------------------------------------ #
    # Average accuracy across all domains                                  #
    # ------------------------------------------------------------------ #
    domain_avg = perf_game.groupby("domain_key")[METRICS].mean().round(3)
    domain_avg.columns = [m.capitalize() + " Acc." for m in METRICS]
    domain_avg = domain_avg.reindex(sort_domain_key_index(domain_avg.index))
    domain_counts = perf_game.groupby("domain_key").size()
    weighted_avg = (
        domain_avg.multiply(domain_counts.reindex(domain_avg.index), axis=0).sum()
        / domain_counts.sum()
    ).round(3)
    domain_avg.loc["Weighted Avg."] = weighted_avg
    print_table("Average Accuracy by Domain", domain_avg)

    # ------------------------------------------------------------------ #
    # Optional CSV export                                                  #
    # ------------------------------------------------------------------ #
    if output:
        combined = (
            df.groupby(["domain_key", "game"])[METRICS].mean().round(3).reset_index()
        )
        combined.rename(columns={"domain_key": "domain"}, inplace=True)
        combined.to_csv(output, index=False)
        print(f"Detailed results saved to: {output}")


if __name__ == "__main__":
    main()
