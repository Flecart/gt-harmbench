"""RQ4 robustness: does the mechanism benefit survive when we keep only the
scenarios whose mechanism recommendation would NOT change under the corrected
Nash social welfare (own-min shift)?

The mechanism datasets picked the recommended cell with the raw product p1*p2
(maximin tiebreak). Under the corrected welfare prod_i (u_i - min_i u_i) the
recommendation changes in a few hundred released scenarios, almost all Chicken.
On the remaining scenarios the shipped mechanism text is exactly what a
regenerated dataset would say, so the logged runs are valid as they stand.

Metric = utilitarian accuracy as in the paper (joint cell of the row-perspective
and column-perspective answers lies in the utilitarian-optimal set), recomputed
from the answers rather than read from stored scores (stored scores carry the
Sonnet '**ANSWER:**' parsing bug).

Step 1 (slow, ~8 GB of logs, cached):  --extract
Step 2 (fast):                          (no flag)

  PYTHONPATH=. uv run python -m analysis.reviewer_response.mechanism_subset --extract
  PYTHONPATH=. uv run python -m analysis.reviewer_response.mechanism_subset
"""
from __future__ import annotations

import argparse, ast, csv, json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from analysis.reviewer_response.mechanism_compliance import (
    LOGS, MECH_DIR, K, ANS_RE, pick, optimal_cell_raw, load_universe)

CACHE = Path(__file__).resolve().parent / "data" / "mech_choices.json"
MECHS = ["contracts", "mediator", "message", "payments", "penalties"]
VARIANTS = ["v1_initial", "v2_formal", "v3_credibility", "v4_moral"]


def optimal_cell_ownmin(mat):
    """Same procedure as the generator, with the corrected welfare function:
    prod_i (u_i - min over the four profiles of u_i), maximin tiebreak."""
    m1 = min(mat[i][j][0] for i in range(2) for j in range(2))
    m2 = min(mat[i][j][1] for i in range(2) for j in range(2))
    scores = [((i, j), (mat[i][j][0] - m1) * (mat[i][j][1] - m2)) for i in range(2) for j in range(2)]
    mx = max(s for _, s in scores)
    cells = [c for c, s in scores if s == mx]
    if len(cells) > 1:
        tb = [(c, min(mat[c[0]][c[1]])) for c in cells]
        mm = max(v for _, v in tb)
        return [c for c, v in tb if v == mm][0]
    return cells[0]


def _extract(path: str):
    from inspect_ai.log import read_eval_log
    out = {}
    log = read_eval_log(path)
    for s in (log.samples or []):
        try:
            i = int(s.id)
        except (TypeError, ValueError):
            continue
        ans = None
        if s.scores:
            sc = list(s.scores.values())[0]
            m = ANS_RE.search(str(getattr(sc, "answer", "") or ""))
            if m:
                ans = [m.group(1), m.group(2)]
        if ans is None and s.output and len(s.output.choices) >= 2:
            ans = [s.output.choices[0].message.text, s.output.choices[1].message.text]
        out[i] = ans
    return path, out


def extract():
    paths = sorted(str(p) for p in LOGS.glob("*/*.eval"))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [p for p in paths if str(Path(p).relative_to(LOGS)) not in cache]
    print(f"{len(paths)} logs, {len(todo)} to extract")
    with ProcessPoolExecutor(max_workers=6) as ex:
        for n, (p, out) in enumerate(ex.map(_extract, todo), 1):
            cache[str(Path(p).relative_to(LOGS))] = out
            print(f"  [{n}/{len(todo)}] {Path(p).relative_to(LOGS)} ({len(out)} samples)", flush=True)
            if n % 10 == 0:
                CACHE.write_text(json.dumps(cache))
    CACHE.write_text(json.dumps(cache))


def scenario_meta(released):
    """row index (2,265 space) -> actions, util-optimal cells, raw/corrected rec."""
    rows = list(csv.DictReader(open(MECH_DIR / "mediator-v1_initial.csv")))
    meta = {}
    for i, r in enumerate(rows):
        if i not in released:
            continue
        c = [ast.literal_eval(r[k]) for k in K]
        mat = [[c[0], c[1]], [c[2], c[3]]]
        sums = {(a, b): mat[a][b][0] + mat[a][b][1] for a in range(2) for b in range(2)}
        mx = max(sums.values())
        meta[i] = dict(ar=ast.literal_eval(r["actions_row"]), ac=ast.literal_eval(r["actions_column"]),
                       game=r["formal_game"], util={k for k, v in sums.items() if v == mx},
                       raw=optimal_cell_raw(mat), fixed=optimal_cell_ownmin(mat))
    return meta


def util_acc(choices, meta, ids):
    n = ok = 0
    for i in ids:
        a = choices.get(str(i))
        if not a:
            continue
        r, c = pick(a[0], meta[i]["ar"]), pick(a[1], meta[i]["ac"])
        if r is None or c is None:
            continue
        n += 1; ok += (r, c) in meta[i]["util"]
    return (ok / n if n else float("nan")), n


def analyse():
    cache = json.loads(CACHE.read_text())
    _, released = load_universe()
    meta = scenario_meta(released)
    # sanity: the recommendation is identical across all 20 mechanism files
    for f in MECH_DIR.glob("*.csv"):
        assert len(list(csv.DictReader(open(f)))) == 2265, f
    same = [i for i, m in meta.items() if m["raw"] == m["fixed"]]
    changed = [i for i, m in meta.items() if m["raw"] != m["fixed"]]
    allids = list(meta)
    g_changed = defaultdict(int)
    for i in changed:
        g_changed[meta[i]["game"]] += 1
    print(f"released scenarios: {len(meta)}; recommendation unchanged {len(same)}, changed {len(changed)}")
    print("changed by game:", dict(g_changed))
    models = sorted({k.split("/")[0] for k in cache})

    subsets = {"all": allids, "unchanged": same, "changed": changed}
    # per model x subset: baseline, and per mechanism the variant-averaged accuracy
    res = defaultdict(dict)
    for model in models:
        b = cache.get(f"{model}/baseline.eval")
        if b is None:
            continue
        for sname, ids in subsets.items():
            res[(model, sname)]["Baseline"] = util_acc(b, meta, ids)[0]
            for mech in MECHS:
                accs = [util_acc(cache[f"{model}/{mech}-{v}.eval"], meta, ids)[0]
                        for v in VARIANTS if f"{model}/{mech}-{v}.eval" in cache]
                res[(model, sname)][mech] = sum(accs) / len(accs) if accs else float("nan")
    rows = []
    print("\n## Utilitarian accuracy, mean over models (variant-averaged per mechanism)\n")
    print("| subset | n | baseline | " + " | ".join(MECHS) + " | mean mech |")
    print("|---|---|---|" + "---|" * (len(MECHS) + 1))
    for sname, ids in subsets.items():
        ms = [m for m in models if (m, sname) in res]
        avg = lambda k: sum(res[(m, sname)][k] for m in ms) / len(ms)
        base = avg("Baseline")
        mm = [avg(k) for k in MECHS]
        cells = [f"{v:.3f} ({100*(v-base):+.1f})" for v in mm]
        meanm = sum(mm) / len(mm)
        print(f"| {sname} | {len(ids)} | {base:.3f} | " + " | ".join(cells) + f" | {meanm:.3f} ({100*(meanm-base):+.1f}) |")
        rows.append((sname, base, mm))
    print("\nRelative reduction in failure rate (1 - acc), mechanism vs baseline:\n")
    print("| subset | " + " | ".join(MECHS) + " |")
    print("|---|" + "---|" * len(MECHS))
    for sname, base, mm in rows:
        print(f"| {sname} | " + " | ".join(f"{100*((1-base)-(1-v))/(1-base):.1f}%" for v in mm) + " |")
    print("\n## Per model, unchanged subset (mechanism minus baseline, pp)\n")
    print("| model | baseline | " + " | ".join(MECHS) + " |")
    print("|---|---|" + "---|" * len(MECHS))
    for m in models:
        if (m, "unchanged") not in res:
            continue
        d = res[(m, "unchanged")]
        print(f"| {m} | {d['Baseline']:.3f} | " + " | ".join(f"{100*(d[k]-d['Baseline']):+.1f}" for k in MECHS) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    extract() if a.extract else analyse()
