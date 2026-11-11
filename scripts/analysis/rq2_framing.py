"""
RQ2 framing study: compare abstract vs. narrative eval logs.

Scans two log directories, matches logs by model name, and reports:
  1. Per-game accuracy table (both conditions side by side)
  2. GT-HarmBench-weighted aggregate (reweights to match full dataset distribution)

Usage:
    uv run python scripts/analysis/rq2_framing.py \
        --abstract-dir logs/abstract \
        --narrative-dir logs/baseline
"""

import argparse
import glob
import os

import pandas as pd
from inspect_ai.log import read_eval_log

# Canonical game name mapping (logs may use non-canonical casing)
CANONICAL = {
    "prisoner's dilemma": "Prisoner's Dilemma",
    "chicken":            "Chicken",
    "stag hunt":          "Stag Hunt",
    "stag hunt":          "Stag Hunt",
    "coordination":       "Coordination",
    "bach or stravinski": "Bach or Stravinski",
    "no conflict":        "No Conflict",
}

GT_COUNTS = {
    "Prisoner's Dilemma": 490,
    "Chicken":            379,
    "Stag Hunt":          317,
    "Coordination":       180,
    "Bach or Stravinski": 141,
    "No Conflict":         28,
}
GT_TOTAL  = sum(GT_COUNTS.values())
GT_WEIGHTS = {g: n / GT_TOTAL for g, n in GT_COUNTS.items()}

METRICS = ["utilitarian", "rawlsian", "nash", "avg_utilitarian_score", "avg_rawlsian_score", "avg_nash_social_welfare_score"]
DISPLAY = {
    "utilitarian":                    "Util acc",
    "rawlsian":                       "Rawls acc",
    "nash":                           "Nash acc",
    "avg_utilitarian_score":          "Avg util",
    "avg_rawlsian_score":             "Avg Rawls",
    "avg_nash_social_welfare_score":  "Avg NSW",
}


def canonical_game(name: str) -> str:
    return CANONICAL.get(name.strip().lower(), name.strip())


def extract_model_name(log) -> str:
    """Return a short, provider-stripped model identifier."""
    model = log.eval.model if log.eval else "unknown"
    return model.split("/")[-1]


def load_dir(directory: str) -> dict[str, pd.DataFrame]:
    """Return {model_name: per-sample DataFrame} for all .eval files in directory."""
    results = {}
    for path in sorted(glob.glob(os.path.join(directory, "**/*.eval"), recursive=True)):
        try:
            log = read_eval_log(path)
            if not log.samples:
                continue
            model = extract_model_name(log)
            rows = []
            for s in log.samples:
                game = canonical_game(s.metadata.get("formal_game", "unknown"))
                scores = s.scores.get("all_strategies_scorer", {})
                if not scores:
                    continue
                v = scores.value if hasattr(scores, "value") else scores
                row = {"game": game}
                row.update({m: v.get(m, float("nan")) for m in METRICS})
                rows.append(row)
            if rows:
                # If multiple logs for same model, prefer the largest/latest
                df = pd.DataFrame(rows)
                if model not in results or len(df) > len(results[model]):
                    results[model] = df
        except Exception as e:
            print(f"  Warning: could not read {os.path.basename(path)}: {e}")
    return results


def per_game_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("game")[METRICS].mean()


def weighted_agg(pg: pd.DataFrame) -> dict:
    """Apply GT-HarmBench weights; all 6 games must be present."""
    out = {}
    for m in METRICS:
        total = weight_sum = 0.0
        for game, w in GT_WEIGHTS.items():
            if game in pg.index:
                total      += w * pg.loc[game, m]
                weight_sum += w
        out[m] = total / weight_sum if weight_sum > 0 else float("nan")
    return out


def print_per_game_table(model: str, narr_pg: pd.DataFrame, abs_pg: pd.DataFrame) -> None:
    print(f"\n{'─'*80}")
    print(f"  {model}")
    print(f"{'─'*80}")
    header = f"  {'Game':<22}  {'Narrative':>10}  {'Abstract':>10}  {'Δ':>8}"
    for m in ["utilitarian", "avg_utilitarian_score", "nash"]:
        print(f"\n  [{DISPLAY[m]}]")
        print(header)
        for game in GT_COUNTS:
            if game not in narr_pg.index or game not in abs_pg.index:
                continue
            n = narr_pg.loc[game, m]
            a = abs_pg.loc[game, m]
            w = GT_COUNTS[game] / GT_TOTAL
            print(f"  {game:<22}  {n:>10.3f}  {a:>10.3f}  {a-n:>+8.3f}  (w={w:.3f})")


def print_weighted_table(model: str, narr_w: dict, abs_w: dict) -> None:
    print("\n  Weighted aggregate (GT-HarmBench distribution):")
    print(f"  {'Metric':<28}  {'Narrative':>10}  {'Abstract':>10}  {'Δ':>8}")
    print(f"  {'-'*62}")
    for m in METRICS:
        print(f"  {DISPLAY[m]:<28}  {narr_w[m]:>10.3f}  {abs_w[m]:>10.3f}  {abs_w[m]-narr_w[m]:>+8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 framing comparison: abstract vs. narrative")
    parser.add_argument("--abstract-dir",  required=True,  help="Directory with abstract eval logs")
    parser.add_argument("--narrative-dir", required=True,  help="Directory with narrative eval logs")
    args = parser.parse_args()

    print(f"Loading abstract logs from:  {args.abstract_dir}")
    abstract  = load_dir(args.abstract_dir)
    print(f"Loading narrative logs from: {args.narrative_dir}")
    narrative = load_dir(args.narrative_dir)

    matched   = sorted(set(abstract) & set(narrative))
    abs_only  = sorted(set(abstract) - set(narrative))
    narr_only = sorted(set(narrative) - set(abstract))

    print(f"\nMatched models ({len(matched)}): {', '.join(matched) or 'none'}")
    if abs_only:
        print(f"Abstract only (no narrative pair): {', '.join(abs_only)}")
    if narr_only:
        print(f"Narrative only (no abstract pair): {', '.join(narr_only)}")

    for model in matched:
        narr_pg = per_game_means(narrative[model])
        abs_pg  = per_game_means(abstract[model])
        narr_w  = weighted_agg(narr_pg)
        abs_w   = weighted_agg(abs_pg)
        print_per_game_table(model, narr_pg, abs_pg)
        print_weighted_table(model, narr_w, abs_w)


if __name__ == "__main__":
    main()
