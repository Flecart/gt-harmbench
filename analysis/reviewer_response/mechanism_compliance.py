"""Do models follow the mechanism's recommendation — including when it is wrong?

`scripts/generation/generate_mechanism_datasets.py:26` chooses the action each
mechanism recommends with `welfare = p1 * p2` on RAW payoffs (maximin tiebreak).
Because the raw Nash product rewards mutual-harm outcomes, in 341 of the 1,535
released scenarios the mechanism recommends the action leading to a cell where
BOTH players are strictly negative — while a both-non-negative cell exists.

This asks the question that decides how much that matters: do models comply?

For every mechanism run we compute
  - compliance: did the model choose the recommended action?
  - split by whether the recommendation is HARMFUL (points at a both-negative
    cell while a non-negative one was available) or SANE
  - the baseline counterfactual: on the same scenarios, how often does the model
    choose that same action WITHOUT the mechanism (baseline.eval)?

The difference between the mechanism rate and the baseline rate on the harmful
subset is the quantity of interest: how far the misdirected mechanism actually
pushed models toward the catastrophic outcome.

Joins (all verified):
  - `data/mech_variants/*.csv` are index-aligned with
    `data/gt-harmbench-with-targets.csv` (2,265 rows; payoffs and actions
    identical row-for-row), and log `sample.id` is that row index.
  - the released 1,535 map into the 2,265 universe by exact `story_row` match
    (1,535/1,535). Everything below is restricted to that subset.

Read-only. Usage:
  PYTHONPATH=. uv run python -m analysis.reviewer_response.mechanism_compliance
  ... --mechanisms mediator-v1_initial,contracts-v1_initial
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from inspect_ai.log import read_eval_log  # noqa: E402

LOGS = REPO / "logs" / "Mech-Design"
MECH_DIR = REPO / "data" / "mech_variants"
CLEAN = REPO / "data" / "gt-harmbench-cleaned.csv"
WT = REPO / "data" / "gt-harmbench-with-targets.csv"
K = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]
ANS_RE = re.compile(r"Row choice:\s*(.*?),\s*Column choice:\s*(.*)", re.I | re.S)


def norm(x):
    return re.sub(r"[*`]+", "", str(x)).strip().lower().strip(" .:\n\"'")


def pick(choice, actions):
    """Map a raw answer string to an action index by longest-prefix match."""
    c = norm(choice)
    best, blen = None, -1
    for i, a in enumerate(actions):
        na = norm(a)
        if (c == na or c.startswith(na) or na in c) and len(na) > blen:
            best, blen = i, len(na)
    return best


def optimal_cell_raw(mat):
    """Verbatim reimplementation of generate_mechanism_datasets.find_optimal_welfare_cell."""
    scores = [((i, j), mat[i][j][0] * mat[i][j][1]) for i in range(2) for j in range(2)]
    mx = max(s for _, s in scores)
    cells = [c for c, s in scores if s == mx]
    if len(cells) > 1:
        tb = [(c, min(mat[c[0]][c[1]])) for c in cells]
        mm = max(v for _, v in tb)
        return [c for c, v in tb if v == mm][0]
    return cells[0]


def load_universe():
    wt = list(csv.DictReader(open(WT)))
    cl = list(csv.DictReader(open(CLEAN)))
    idx = {r["story_row"]: i for i, r in enumerate(wt)}
    released = {idx[r["story_row"]] for r in cl if r["story_row"] in idx}
    return wt, released


def build_meta(mech_csv, released):
    """row index -> dict(actions, mat, rec_cell, harmful)"""
    rows = list(csv.DictReader(open(mech_csv)))
    meta = {}
    for i, r in enumerate(rows):
        if i not in released:
            continue
        cells = [ast.literal_eval(r[k]) for k in K]
        mat = [[cells[0], cells[1]], [cells[2], cells[3]]]
        rc = optimal_cell_raw(mat)
        p = mat[rc[0]][rc[1]]
        avail = any(c[0] >= 0 and c[1] >= 0 for c in cells)
        meta[i] = {
            "ar": ast.literal_eval(r["actions_row"]),
            "ac": ast.literal_eval(r["actions_column"]),
            "mat": mat, "rec": rc, "game": r["formal_game"],
            "harmful": p[0] < 0 and p[1] < 0 and avail,
        }
    return meta


def read_choices(path):
    """row index -> (row_idx, col_idx) or None"""
    out = {}
    log = read_eval_log(str(path))
    for s in (log.samples or []):
        try:
            i = int(s.id)
        except (TypeError, ValueError):
            continue
        ans = None
        if s.scores:
            sc = list(s.scores.values())[0]
            if getattr(sc, "answer", None):
                m = ANS_RE.search(str(sc.answer))
                if m:
                    ans = (m.group(1), m.group(2))
        if ans is None and s.output and len(s.output.choices) >= 2:
            ans = (s.output.choices[0].message.text,
                   s.output.choices[1].message.text)
        out[i] = ans
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanisms", default="mediator-v1_initial")
    args = ap.parse_args()
    mechs = [m.strip() for m in args.mechanisms.split(",") if m.strip()]

    _, released = load_universe()
    models = sorted(p.name for p in LOGS.iterdir() if p.is_dir())

    print("# Do models follow the mechanism — including when it points at harm?\n")
    print(f"Restricted to the {len(released)} released scenarios.\n")

    for mech in mechs:
        meta = build_meta(MECH_DIR / f"{mech}.csv", released)
        harmful = {i for i, m in meta.items() if m["harmful"]}
        print(f"## {mech}\n")
        print(f"Recommendation points at a both-negative cell in "
              f"**{len(harmful)}/{len(meta)}** released scenarios.\n")
        print("`comply` = model chose the recommended action (row perspective). "
              "`baseline` = same action, same scenarios, with no mechanism.\n")
        print("| model | comply (sane) | comply (harmful) | baseline (harmful) "
              "| mechanism-induced shift |")
        print("|---|---|---|---|---|")
        agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
        for model in models:
            mp = LOGS / model / f"{mech}.eval"
            bp = LOGS / model / "baseline.eval"
            if not mp.exists() or not bp.exists():
                continue
            try:
                mc, bc = read_choices(mp), read_choices(bp)
            except Exception as e:                       # noqa: BLE001
                print(f"| {model} | (log unreadable: {type(e).__name__}) | | | |")
                continue
            ns = cs = nh = ch = nb = cb = 0
            for i, m in meta.items():
                rec_r = m["rec"][0]
                a = mc.get(i)
                if a:
                    k = pick(a[0], m["ar"])
                    if k is not None:
                        if m["harmful"]:
                            nh += 1; ch += (k == rec_r)
                        else:
                            ns += 1; cs += (k == rec_r)
                if m["harmful"]:
                    b = bc.get(i)
                    if b:
                        k = pick(b[0], m["ar"])
                        if k is not None:
                            nb += 1; cb += (k == rec_r)
            g = agg[mech]
            g[0] += cs; g[1] += ns; g[2] += ch; g[3] += nh; g[4] += cb; g[5] += nb
            f = lambda a, b: f"{100*a/b:.1f}% ({b})" if b else "—"
            shift = (f"{100*(ch/nh - cb/nb):+.1f} pp" if nh and nb else "—")
            print(f"| {model} | {f(cs,ns)} | {f(ch,nh)} | {f(cb,nb)} | {shift} |")
        g = agg[mech]
        f = lambda a, b: f"{100*a/b:.1f}%" if b else "—"
        shift = (f"{100*(g[2]/g[3] - g[4]/g[5]):+.1f} pp" if g[3] and g[5] else "—")
        print(f"| **pooled** | **{f(g[0],g[1])}** | **{f(g[2],g[3])}** | "
              f"**{f(g[4],g[5])}** | **{shift}** |")
        print()


if __name__ == "__main__":
    main()
