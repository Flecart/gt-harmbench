"""Patch plan for the scenarios whose off-diagonal mirror is broken.

A symmetric 2x2 game requires cell(2,1) to be the reflection of cell(1,2):
if (1,2) pays [a,b] then (2,1) must pay [b,a]. In 14 of the 1,535 cleaned
scenarios that fails. There are exactly two minimal repairs:

  A   overwrite (2,1) with the reflection of (1,2)
  B   overwrite (1,2) with the reflection of (2,1)

For each scenario this reports, for both directions: whether the repaired game
still passes its assigned class's structural verifier, and whether each of the
four benchmark targets is preserved. A repair is SAFE when the class holds and
every target is unchanged — then the scenario needs no re-labelling of its
targets and the only consequence is that models must be re-queried on the new
matrix. Ties are broken toward the smaller edit (L1 distance).

Read-only: prints a plan, writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.reviewer_response.lib import load_instances  # noqa: E402
from analysis.reviewer_response.ordinal_asymmetry_audit import (  # noqa: E402
    sym_up_to_relabel,
)
from scripts.generation.fix_contextualizations import is_valid_game  # noqa: E402
from game.utils import (  # noqa: E402
    find_nash_equilibria, find_utility_maximizing, find_Rawlsian_actions,
    find_nash_social_welfare,
)

CELL_KEYS = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]


def mirrored(cells) -> bool:
    return cells[1] == cells[2][::-1]


def as_matrix(cells):
    return [[cells[0], cells[1]], [cells[2], cells[3]]]


def payoffs_dict(cells):
    return {k: cells[i] for i, k in enumerate(CELL_KEYS)}


def targets(cells):
    m = as_matrix(cells)
    def norm(x):
        try:
            return frozenset(map(str, x)) if isinstance(x, (list, set, tuple)) else str(x)
        except TypeError:
            return str(x)
    return {
        "nash": norm(find_nash_equilibria(m)),
        "util": norm(find_utility_maximizing(m)),
        "rawls": norm(find_Rawlsian_actions(m)),
        "nsw": norm(find_nash_social_welfare(m)),
    }


def repair(cells, direction):
    c = [list(x) for x in cells]
    if direction == "A":            # (2,1) := reflect (1,2)
        c[2] = [c[1][1], c[1][0]]
    else:                           # (1,2) := reflect (2,1)
        c[1] = [c[2][1], c[2][0]]
    return c


def l1(a, b):
    return sum(abs(x - y) for ca, cb in zip(a, b) for x, y in zip(ca, cb))


def main():
    insts = load_instances()
    broken = [i for i in insts
              if not sym_up_to_relabel(i.cells) and not mirrored(i.cells)]

    print("# Mirror-repair plan\n")
    print(f"Scenarios with a broken off-diagonal mirror: **{len(broken)}**\n")
    print("`class` = repaired game still passes its assigned verifier. "
          "`targets` = all four benchmark targets (Nash, utilitarian, Rawlsian, "
          "NSW) unchanged by the repair. `edit` = L1 distance from the original "
          "matrix. **SAFE** = class holds and targets are preserved.\n")
    print("| id | class | original (1,2) / (2,1) | A: (2,1):=refl(1,2) | A class | "
          "A targets | A edit | B: (1,2):=refl(2,1) | B class | B targets | B edit | pick |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")

    n_safe_both = n_safe_one = n_safe_none = 0
    picks = {}
    for it in broken:
        base_t = targets(it.cells)
        row = [it.id, it.game, f"{it.cells[1]} / {it.cells[2]}"]
        results = {}
        for d in ("A", "B"):
            rc = repair(it.cells, d)
            ok_cls = bool(is_valid_game(it.game, payoffs_dict(rc)))
            ok_tgt = targets(rc) == base_t
            results[d] = (rc, ok_cls, ok_tgt, l1(rc, it.cells))
            newcell = rc[2] if d == "A" else rc[1]
            row += [str(newcell), "yes" if ok_cls else "**no**",
                    "same" if ok_tgt else "**CHANGED**", str(results[d][3])]
        safe = [d for d in ("A", "B") if results[d][1] and results[d][2]]
        if len(safe) == 2:
            n_safe_both += 1
            pick = min(safe, key=lambda d: results[d][3])
            label = f"{pick} (smaller edit)"
        elif len(safe) == 1:
            n_safe_one += 1
            pick = safe[0]
            label = f"**{pick} (only safe option)**"
        else:
            n_safe_none += 1
            pick = None
            label = "**NEITHER — needs review**"
        picks[it.id] = pick
        row.append(label)
        print("| " + " | ".join(row) + " |")

    print()
    print(f"- both repairs safe: **{n_safe_both}/{len(broken)}**")
    print(f"- exactly one safe : **{n_safe_one}/{len(broken)}**")
    print(f"- neither safe     : **{n_safe_none}/{len(broken)}**")
    if n_safe_none == 0:
        print("\n> Every scenario has at least one repair that preserves both its "
              "game class and all four targets, so no target column needs "
              "relabelling.")
    print(f"\nAffected ids: {', '.join(i.id for i in broken)}")
    print("\nRe-query cost for the MAIN evaluation: **none**. The main-eval prompt "
          "(eval/eval.py:52-76) contains only the narrative and the two action "
          "labels — the payoff matrix is never shown — so patching cannot change "
          "any model's input, and the targets are unchanged, so no score moves "
          "either. The patch is a fix to the released dataset only.\n"
          "The RQ2 abstract condition is the exception: "
          "data/gt-harmbench-abstract-full.csv renders the payoffs into the prompt "
          "text, and carries the same defective matrices under the same ids. "
          f"Patching there needs those rows regenerated plus {len(broken)} x 15 "
          f"models x 2 roles = {len(broken)*15*2} re-queries.")


if __name__ == "__main__":
    main()
