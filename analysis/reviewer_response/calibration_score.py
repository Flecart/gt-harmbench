"""Score the round-2 calibration (or core) annotations.

    uv run python analysis/reviewer_response/calibration_score.py                 # calibration set
    uv run python analysis/reviewer_response/calibration_score.py --set core      # core 60, r2 files

Reads every annotation_<name>_cal.csv (or annotation_<name>_r2.csv), reports
per-category raw agreement, Cohen's kappa (expected agreement from the raters'
marginals -- never 1/6), Gwet's AC1 and PABAK between each pair of annotators,
and each annotator against the three-judge majority from the key. Writes a
disagreement sheet with every rater's quote side by side for the discussion.
"""
from __future__ import annotations

import argparse, csv, itertools
from collections import Counter
from pathlib import Path

HV = Path(__file__).resolve().parent / "human_validation"
CATS = ["NASH_DOMINANT", "PAYOFF_MAX", "UTILITARIAN", "RAWLSIAN",
        "CATASTROPHE_PREVENTION", "PRECAUTIONARY", "AI_SAFETY", "DOMAIN_OTHER"]
JUDGES = ["sonnet", "gpt", "grok"]


def stats(x: list[int], y: list[int]) -> dict:
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    pi = (px + py) / 2
    ac1_e = 2 * pi * (1 - pi)
    return {"raw": po,
            "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"),
            "ac1": (po - ac1_e) / (1 - ac1_e) if ac1_e < 1 else float("nan"),
            "pabak": 2 * po - 1}


def load_annotations(pattern: str, id_col: str) -> dict[str, dict]:
    raters = {}
    for p in sorted(HV.glob(pattern)):
        name = p.stem.replace("annotation_", "").rsplit("_", 1)[0]
        rows = {}
        for r in csv.DictReader(open(p)):
            rows[r["id"]] = {c: {"present": int(r.get(c) or 0), "use": r.get(f"{c}__use", ""),
                                 "quote": r.get(f"{c}__quote", "")} for c in CATS}
        raters[name] = rows
        print(f"{name}: {len(rows)} traces labelled ({p.name})")
    return raters


def load_key(path: Path) -> dict[str, dict]:
    key = {}
    for r in csv.DictReader(open(path)):
        key[r["cal_id"]] = {c: {"votes": sum(int(r[f"{c}__{j}"]) for j in JUDGES),
                                "quotes": {j: r.get(f"{c}__{j}_quote", "") for j in JUDGES}}
                            for c in CATS}
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["calibration", "core"], default="calibration")
    a = ap.parse_args()
    if a.set == "calibration":
        raters = load_annotations("calibration/annotation_*_cal.csv", "cal_id")
        key = load_key(HV.parent / "data/human_validation_keys/_calibration_key.csv")
        sheet = HV / "calibration/disagreements.csv"
    else:
        raters = load_annotations("annotation_*_r2.csv", "annotation_id")
        key = None
        sheet = HV / "disagreements_r2.csv"
    if not raters:
        print("no annotation files found"); return

    common = sorted(set.intersection(*(set(r) for r in raters.values())))
    print(f"\ncommon traces: {len(common)}")
    names = list(raters)
    if key:
        raters["JUDGE_MAJORITY"] = {i: {c: {"present": int(key[i][c]["votes"] >= 2), "use": "", "quote": ""}
                                        for c in CATS} for i in common}
    pairs = list(itertools.combinations(list(raters), 2))

    print(f"\n{'category':24s}" + "".join(f"{n[:10]:>11s}" for n in raters) + "   " +
          "".join(f"{(x[:5]+'|'+y[:5]):>22s}" for x, y in pairs))
    print(f"{'(prevalence)':24s}{'':{11*len(raters)}s}   " + "".join(f"{'raw  κ   AC1  PABAK':>22s}" for _ in pairs))
    pooled = {p: ([], []) for p in pairs}
    for c in CATS:
        row = f"{c:24s}" + "".join(f"{sum(raters[n][i][c]['present'] for i in common)/len(common):11.2f}" for n in raters) + "   "
        for x, y in pairs:
            xs = [raters[x][i][c]["present"] for i in common]; ys = [raters[y][i][c]["present"] for i in common]
            pooled[(x, y)][0].extend(xs); pooled[(x, y)][1].extend(ys)
            s = stats(xs, ys)
            row += f"{s['raw']:5.2f}{s['kappa']:5.2f}{s['ac1']:6.2f}{s['pabak']:6.2f}"
        print(row)
    row = f"{'POOLED':24s}{'':{11*len(raters)}s}   "
    for p in pairs:
        s = stats(*pooled[p]); row += f"{s['raw']:5.2f}{s['kappa']:5.2f}{s['ac1']:6.2f}{s['pabak']:6.2f}"
    print(row)

    # 'use' distribution per human rater -- shows whether someone never uses 'mention'
    print("\nhow each annotator used the three-way tag (over all ticked boxes):")
    for n in names:
        cnt = Counter(raters[n][i][c]["use"] for i in common for c in CATS if raters[n][i][c]["use"])
        print(f"  {n:12s} {dict(cnt)}")

    # disagreement sheet: any trace x category where the humans differ, with everyone's quote
    with open(sheet, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["id", "category"] + [f"{n}" for n in names] + [f"{n}_use" for n in names] + [f"{n}_quote" for n in names]
        if key: hdr += ["judge_votes"] + [f"{j}_quote" for j in JUDGES]
        w.writerow(hdr)
        n_rows = 0
        for i in common:
            for c in CATS:
                vals = [raters[n][i][c]["present"] for n in names]
                if len(set(vals)) > 1:
                    row = [i, c] + vals + [raters[n][i][c]["use"] for n in names] + [raters[n][i][c]["quote"] for n in names]
                    if key: row += [key[i][c]["votes"]] + [key[i][c]["quotes"][j] for j in JUDGES]
                    w.writerow(row); n_rows += 1
    print(f"\n{n_rows} human disagreements -> {sheet}")


if __name__ == "__main__":
    main()
