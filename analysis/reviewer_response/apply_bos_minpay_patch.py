"""Equalise the miscoordination payoffs of the 2 BoS scenarios (408, 1233) whose
players have different worst payoffs. Third step of the symmetry correction;
run after apply_mirror_patch.py and apply_bos_tie_patch.py.

Why. Both scenarios are ordinally symmetric, so the earlier ordinal audit passed
them, but they are not cardinally symmetric: in 408 the miscoordination cells are
(2,1) and (3,2), so the row player's worst payoff is 2 and the column player's is
1. Nash social welfare measures each payoff from the player's own worst payoff,
so this one-unit asymmetry breaks the tie between the two coordination cells
(48 vs 45 in 408), which every other welfare function, and the game's own
symmetry, treat as tied. No other released scenario is affected.

Fix. Same rule as apply_bos_tie_patch.py: set both miscoordination cells to
[m, m], m = the smallest payoff appearing in them.

Verified before writing: still a valid BoS; Nash equilibria, utilitarian and
Rawlsian targets unchanged; NSW returns to the two-cell tie. The main eval and
the mechanism conditions never show the payoff matrix, so no model input
changes. RQ2 (abstract/gamified) shows it: exclude these two there, as for the
other 22.

Usage:
  PYTHONPATH=. uv run python -m analysis.reviewer_response.apply_bos_minpay_patch --dry-run
  PYTHONPATH=. uv run python -m analysis.reviewer_response.apply_bos_minpay_patch --write
"""
from __future__ import annotations

import argparse, ast, csv, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.generation.fix_contextualizations import is_valid_game  # noqa: E402
from game.metrics import player_minima  # noqa: E402
from game.utils import (find_nash_equilibria, find_utility_maximizing,  # noqa: E402
                        find_Rawlsian_actions, find_nash_social_welfare)

CSV_PATH = REPO / "data" / "gt-harmbench-cleaned.csv"
K = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]
IDS = {"408", "1233"}


def mat(c):
    return [[c[0], c[1]], [c[2], c[3]]]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(CSV_PATH)))
    fields = list(rows[0].keys())
    # the selection rule, checked: these are exactly the BoS with unequal minima
    # (1092 also has unequal minima, but its coordination cells are not tied in
    # the first place, so NSW and utilitarian already agree; left untouched.)
    def breaks_tie(r):
        m_ = mat([ast.literal_eval(r[k]) for k in K])
        return (len(set(player_minima(m_))) == 2
                and sorted(find_nash_social_welfare(m_)) != sorted(find_utility_maximizing(m_)))
    found = {r["id"] for r in rows if r["formal_game"] == "Bach or Stravinski" and breaks_tie(r)}
    if found - IDS:
        sys.exit(f"unexpected BoS with unequal minima: {sorted(found - IDS)}")
    changed = 0
    for r in rows:
        if r["id"] not in found:
            continue
        cells = [ast.literal_eval(r[k]) for k in K]
        m = min(cells[1] + cells[2])
        new = [cells[0], [m, m], [m, m], cells[3]]
        ok = (bool(is_valid_game("Bach or Stravinski", dict(zip(K, new))))
              and sorted(find_nash_equilibria(mat(new)) or []) == sorted(find_nash_equilibria(mat(cells)) or [])
              and find_utility_maximizing(mat(new)) == find_utility_maximizing(mat(cells))
              and find_Rawlsian_actions(mat(new)) == find_Rawlsian_actions(mat(cells))
              and sorted(find_nash_social_welfare(mat(new))) == sorted(find_utility_maximizing(mat(new))))
        if not ok:
            sys.exit(f"id {r['id']}: verification failed, nothing written")
        print(f"  id {r['id']:>4}  {cells[1]}/{cells[2]}  ->  {new[1]}/{new[2]}  "
              f"NSW {sorted(find_nash_social_welfare(mat(cells)))} -> {sorted(find_nash_social_welfare(mat(new)))}")
        for k, v in zip(K, new):
            r[k] = str(v)
        changed += 1
    print(f"\n{changed} scenarios equalised.")
    if args.write and changed:
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
