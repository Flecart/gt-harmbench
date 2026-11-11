"""Equalise the miscoordination payoffs in the 9 remaining asymmetric BoS.

Second half of the symmetry correction. Run AFTER `apply_mirror_patch.py`.

Background. A symmetric 2x2 game requires whichever cell pair sits on the
diagonal to be internally tied (row payoff == column payoff *within* the cell),
because transposing the players maps a diagonal cell to itself. Battle of the
Sexes has an internally untied diagonal by construction -- (10,7) and (7,10) --
so on the face of it every BoS should fail. It does not: relabelling the column
actions moves the miscoordination pair onto the diagonal, and in the textbook
form that pair IS tied, e.g. (2,2) and (1,1). 132 of the 141 BoS scenarios pass
for exactly this reason.

The 9 that fail are those whose miscoordination cells are ALSO internally untied
-- (0,1) and (1,0) rather than (1,1) and (1,1) -- so neither labelling puts a
tied pair on the diagonal. The asymmetry is one payoff unit in 8 of the 9 and
two in the last. They remain valid BoS instances; they are simply not symmetric
games, which is inconsistent with how the benchmark describes itself.

Fix: set both miscoordination cells to [m, m] where m is the minimum payoff
appearing across the two cells. Tying to the maximum is equally safe (verified);
the minimum is chosen because it never inflates a payoff.

Verified for all 9 before writing:
  - the repaired matrix still passes the BoS structural verifier
  - all four targets (Nash equilibria, utilitarian, Rawlsian, Nash social
    welfare) are unchanged
  - the scenario becomes ordinally symmetric up to relabelling

Together with the 14 mirror repairs this touches 22 distinct scenarios (id 1380
is in both sets) and takes the residual to zero.

Consequences: none for the main evaluation or for the mechanism conditions,
neither of which shows the payoff matrix to the model, and no target moves. The
abstract and gamified conditions do show it, so those 22 are excluded from the
RQ2 comparisons rather than re-queried.

Idempotent: scenarios already symmetric are skipped. Revert both patches with
`git checkout -- data/gt-harmbench-cleaned.csv`.

Usage:
  PYTHONPATH=. uv run python -m analysis.reviewer_response.apply_bos_tie_patch --dry-run
  PYTHONPATH=. uv run python -m analysis.reviewer_response.apply_bos_tie_patch --write
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.reviewer_response.ordinal_asymmetry_audit import sym_up_to_relabel  # noqa: E402
from scripts.generation.fix_contextualizations import is_valid_game  # noqa: E402
from game.utils import (  # noqa: E402
    find_nash_equilibria, find_utility_maximizing, find_Rawlsian_actions,
    find_nash_social_welfare,
)

CSV_PATH = REPO / "data" / "gt-harmbench-cleaned.csv"
K = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]


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
    changed = 0

    for r in rows:
        cells = [ast.literal_eval(r[k]) for k in K]
        if sym_up_to_relabel(cells):
            continue
        if r["formal_game"] != "Bach or Stravinski":
            print(f"  id {r['id']}: unexpected asymmetric {r['formal_game']} — "
                  f"run apply_mirror_patch.py first")
            sys.exit(1)

        m = min(cells[1] + cells[2])
        new = [cells[0], [m, m], [m, m], cells[3]]

        ok_cls = bool(is_valid_game("Bach or Stravinski", dict(zip(K, new))))
        ok_tgt = targets(new) == targets(cells)
        ok_sym = sym_up_to_relabel(new)
        if not (ok_cls and ok_tgt and ok_sym):
            print(f"  id {r['id']} ABORT: class={ok_cls} targets={ok_tgt} "
                  f"symmetric={ok_sym}")
            sys.exit(1)

        print(f"  id {r['id']:>4}  {cells[1]}/{cells[2]}  ->  [{m},{m}]/[{m},{m}]")
        for k, v in zip(K, new):
            r[k] = str(v)
        changed += 1

    print(f"\n{changed} Bach or Stravinski scenarios equalised.")
    if args.write:
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {CSV_PATH}")
    else:
        print("Dry run — nothing written.")


if __name__ == "__main__":
    main()
