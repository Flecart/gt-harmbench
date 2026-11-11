"""Symmetrise the 14 scenarios whose off-diagonal cells are not exact mirrors.

A symmetric 2x2 game requires cell(2,1) to be the reflection of cell(1,2). In 14
of the 1,535 released scenarios that fails. This rewrites one off-diagonal cell
in each, in the direction that does least ordinal damage.

Verified before writing (see mirror_patch_plan.py):
  - the repaired matrix still passes the structural verifier for its assigned
    game class, in all 14 cases and in both directions
  - all four targets (Nash equilibria, utilitarian, Rawlsian, Nash social
    welfare) are unchanged, in all 14 cases and in both directions

Consequences: none for the main evaluation. The main-eval prompt shows only the
narrative and the two action labels, never the payoff matrix, so no model input
changes; and since the targets are unchanged, no score changes. This is a fix to
the released artifact.

Caveat recorded honestly: 6 of the 14 (158, 184, 238, 533, 962, 1529) have
role-asymmetric narratives -- the two players hold different roles and different
action sets, so their payoffs were legitimately asymmetric. Symmetrising them
introduces a mild story/matrix mismatch: one flipped pairwise comparison, or a
tie where the narrative implies a strict ordering. The direction for each was
chosen to minimise exactly that. The other 8 share an identical action set
between the players, where a broken mirror is a genuine generation slip.

Idempotent: rows already mirrored are left alone. Writes in place; revert with
`git checkout -- data/gt-harmbench-cleaned.csv`.

Usage:
  PYTHONPATH=. uv run python -m analysis.reviewer_response.apply_mirror_patch --dry-run
  PYTHONPATH=. uv run python -m analysis.reviewer_response.apply_mirror_patch --write
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.generation.fix_contextualizations import is_valid_game  # noqa: E402
from game.utils import (  # noqa: E402
    find_nash_equilibria, find_utility_maximizing, find_Rawlsian_actions,
    find_nash_social_welfare,
)

CSV_PATH = REPO / "data" / "gt-harmbench-cleaned.csv"
K = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]

# direction A: (2,1) := reflect(1,2);  direction B: (1,2) := reflect(2,1)
# A everywhere except 158, where B costs 0 reversals instead of 1.
PLAN = {"63": "A", "158": "B", "184": "A", "238": "A", "308": "A", "512": "A",
        "533": "A", "658": "A", "750": "A", "962": "A", "1126": "A",
        "1170": "A", "1380": "A", "1529": "A"}


def repair(cells, d):
    c = [list(x) for x in cells]
    if d == "A":
        c[2] = [c[1][1], c[1][0]]
    else:
        c[1] = [c[2][1], c[2][0]]
    return c


def targets(cells):
    m = [[cells[0], cells[1]], [cells[2], cells[3]]]

    def norm(x):
        return frozenset(map(str, x)) if isinstance(x, (list, set, tuple)) else str(x)
    return (norm(find_nash_equilibria(m)), norm(find_utility_maximizing(m)),
            norm(find_Rawlsian_actions(m)), norm(find_nash_social_welfare(m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (args.write or args.dry_run):
        ap.error("pass --dry-run or --write")

    rows = list(csv.DictReader(open(CSV_PATH)))
    fields = list(rows[0].keys())
    changed = skipped = 0

    for r in rows:
        d = PLAN.get(r["id"])
        if not d:
            continue
        cells = [ast.literal_eval(r[k]) for k in K]
        if cells[1] == cells[2][::-1]:
            skipped += 1
            print(f"  id {r['id']:>4}  already mirrored, skipping")
            continue
        new = repair(cells, d)

        ok_cls = bool(is_valid_game(r["formal_game"], dict(zip(K, new))))
        ok_tgt = targets(new) == targets(cells)
        if not (ok_cls and ok_tgt):
            print(f"  id {r['id']:>4}  ABORT: class={ok_cls} targets={ok_tgt}")
            sys.exit(1)

        print(f"  id {r['id']:>4}  {r['formal_game']:<18} dir {d}  "
              f"{cells[1]}/{cells[2]}  ->  {new[1]}/{new[2]}")
        for k, v in zip(K, new):
            r[k] = str(v)
        changed += 1

    print(f"\n{changed} rows repaired, {skipped} already correct.")
    if args.write:
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {CSV_PATH}")
    else:
        print("Dry run - nothing written.")


if __name__ == "__main__":
    main()
