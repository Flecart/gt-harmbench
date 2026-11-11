"""Round-2 human validation: agreement between human annotators and the RQ3 judge
on the core 60 traces. kappa uses each rater's own marginals (judge_pilot.kappa).

  PYTHONPATH=. uv run python -m analysis.reviewer_response.hv_agreement [extra annotation files...]

Annotation files: JSON in Isabel's format ({"annotations": [{"id", CATEGORY: 0/1, ...}]})
or the annotation app's CSV (annotation_id + one 0/1 column per category). By default
it picks up every human_validation/annotation_*.json|csv except round-1 files.
"""
from __future__ import annotations

import csv, json, sys
from itertools import combinations
from pathlib import Path

from analysis.reviewer_response.judge_pilot import kappa, CATS, HV

JUDGE = Path(__file__).resolve().parent / "data/judge_pilot/hv_core60__anthropic_claude-sonnet-5.5.jsonl"
ROUND1 = {"annotation_pepijn.csv", "annotation_angelo.csv", "annotation_template.csv"}


def load(p: Path) -> dict:
    if p.suffix == ".json":
        rows = json.load(open(p))["annotations"]
        return {str(r["id"]): {c: int(r[c]) for c in CATS} for r in rows if r.get("block", "core") == "core"}
    out = {}
    for r in csv.DictReader(open(p)):
        if r.get("block", "core") != "core" or not all(r.get(c, "") != "" for c in CATS):
            continue
        out[str(r.get("annotation_id") or r.get("id"))] = {c: int(float(r[c])) for c in CATS}
    return out


def main(extra):
    raters = {}
    files = extra or sorted(p for p in HV.glob("annotation_*") if p.name not in ROUND1 and p.suffix in (".json", ".csv"))
    for p in files:
        raters[Path(p).stem.replace("annotation_", "").capitalize()] = load(Path(p))
    raters["Sonnet 5.5 (judge)"] = {r["id"]: r["labels"] for r in map(json.loads, open(JUDGE)) if "labels" in r}
    print("raters:", {k: len(v) for k, v in raters.items()})
    for a, b in combinations(raters, 2):
        ids = sorted(set(raters[a]) & set(raters[b]), key=int)
        X = [raters[a][i][c] for i in ids for c in CATS]; Y = [raters[b][i][c] for i in ids for c in CATS]
        po, k = kappa(X, Y)
        per = {c: kappa([raters[a][i][c] for i in ids], [raters[b][i][c] for i in ids])[1] for c in CATS}
        print(f"\n{a} vs {b} (n={len(ids)} traces): pooled kappa {k:.2f}, raw agreement {po:.2f}")
        print("   " + "  ".join(f"{c[:8]} {v:.2f}" for c, v in per.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
