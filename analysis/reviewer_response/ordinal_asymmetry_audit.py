"""Audit of the scenarios that are NOT ordinally symmetric up to relabeling.

The claim in the 1wTm reply — "98.6% ordinally symmetric up to relabeling (the
residual, 22 scenarios, we are correcting)" — had no code behind it in the repo.
This reproduces the flagging and then audits the residual set.

Definitions (these reproduce the published rates exactly: 84.4% cardinal,
90.1% strict ordinal, 98.6% up-to-relabeling, residual 22):

  A 2x2 game is SYMMETRIC when player 2's payoff at outcome (i,j) equals
  player 1's payoff at the transposed outcome (j,i). Cell index order is
  0=(1,1) 1=(1,2) 2=(2,1) 3=(2,2), so transposition swaps cells 1 and 2.

  cardinal symmetry   c[k][1] == c[T(k)][0] for all k
  ordinal symmetry    the row-payoff vector [c[k][0]] and the transposed
                      col-payoff vector [c[T(k)][1]] induce the SAME weak
                      ordering (all 6 pairwise comparisons agree, ties included)
  up to relabeling    ordinal symmetry holds under at least one of the 4
                      independent row/column action relabelings

Outputs a markdown report to stdout.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.reviewer_response.lib import (  # noqa: E402
    load_instances, load_choices, util_argmax_set, GAME_TYPES,
)

TR = [0, 2, 1, 3]                      # transposition on cell indices
CELL_NAME = ["(1,1)", "(1,2)", "(2,1)", "(2,2)"]
PAIRS = list(itertools.combinations(range(4), 2))


def sgn(x):
    return (x > 0) - (x < 0)


def row_vec(cells):
    return [cells[k][0] for k in range(4)]


def colT_vec(cells):
    """Player 2's payoffs indexed by the TRANSPOSED outcome."""
    return [cells[TR[k]][1] for k in range(4)]


def kendall_tau(x, y):
    """Number of discordant pairs out of 6. A single adjacent swap gives 1."""
    return sum(1 for a, b in PAIRS if sgn(x[a] - x[b]) != sgn(y[a] - y[b]))


def ord_sym(cells):
    return kendall_tau(row_vec(cells), colT_vec(cells)) == 0


def relabelings(cells):
    """The 4 independent row/column action relabelings."""
    out = []
    for sr in (0, 1):
        for sc in (0, 1):
            out.append([cells[(i ^ sr) * 2 + (j ^ sc)]
                        for i in (0, 1) for j in (0, 1)])
    return out


def sym_up_to_relabel(cells):
    return any(ord_sym(c) for c in relabelings(cells))


def min_tau(cells):
    return min(kendall_tau(row_vec(c), colT_vec(c)) for c in relabelings(cells))


def rank_str(vals):
    """Competition ranking, 1 = best, ties share a rank. Rendered per cell."""
    order = sorted(set(vals), reverse=True)
    return "/".join(str(order.index(v) + 1) for v in vals)


def symmetrize(cells):
    """Nearest symmetric game: orthogonal projection onto the symmetric
    subspace, i.e. average each payoff with its transposed counterpart."""
    return [[(cells[k][0] + cells[TR[k]][1]) / 2,
             (cells[k][1] + cells[TR[k]][0]) / 2] for k in range(4)]


def totals(cells):
    return [c[0] + c[1] for c in cells]


def main():
    insts = load_instances()
    n = len(insts)
    flagged = [it for it in insts if not sym_up_to_relabel(it.cells)]
    keep_ids = {it.id for it in insts} - {it.id for it in flagged}

    print("# Audit — scenarios not ordinally symmetric up to relabeling\n")
    print(f"Reproduced flagging: cardinal "
          f"{sum(all(i.cells[k][1]==i.cells[TR[k]][0] for k in range(4)) for i in insts)}/{n}, "
          f"strict ordinal {sum(ord_sym(i.cells) for i in insts)}/{n}, "
          f"up-to-relabeling {n-len(flagged)}/{n}. "
          f"**Residual: {len(flagged)} scenarios.**\n")

    # ---------------- per-scenario table ----------------
    print("## Per-scenario\n")
    print("`rank` columns are competition ranks (1 = best) for cells "
          "(1,1)/(1,2)/(2,1)/(2,2); P2 is ranked over TRANSPOSED outcomes, so "
          "identical strings mean perfect ordinal symmetry. "
          "`tau` = discordant pairs out of 6 at the given labeling; "
          "`tau*` = minimum over the 4 relabelings (the true distance from "
          "symmetry, since relabeling is permitted).\n")
    print("| id | class | P1 rank | P2 rank | tau | tau* | own-class verifier | "
          "valid classes | util argmax | util argmax (symmetrised) | changed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")

    n_break = n_util_change = 0
    reclass = []
    for it in flagged:
        r, c = row_vec(it.cells), colT_vec(it.cells)
        tau, tstar = kendall_tau(r, c), min_tau(it.cells)
        own = it.game in it.derived
        n_break += not own
        others = [g for g in it.derived if g != it.game]
        u0 = util_argmax_set(totals(it.cells))
        u1 = util_argmax_set(totals(symmetrize(it.cells)))
        changed = u0 != u1
        n_util_change += changed
        reclass.append((it.id, it.game, own, others))
        print(f"| {it.id} | {it.game} | {rank_str(r)} | {rank_str(c)} | {tau} | "
              f"{tstar} | {'pass' if own else '**FAIL**'} | "
              f"{', '.join(others) if others else ('—' if own else '**none**')} | "
              f"{'+'.join(CELL_NAME[i] for i in sorted(u0))} | "
              f"{'+'.join(CELL_NAME[i] for i in sorted(u1))} | "
              f"{'**yes**' if changed else 'no'} |")

    print()
    print(f"- own-class verifier fails: **{n_break}/{len(flagged)}**")
    print(f"- utilitarian argmax changes under symmetrisation: "
          f"**{n_util_change}/{len(flagged)}**")

    print("\n## Reclassification\n")
    only_other = [x for x in reclass if not x[2] and x[3]]
    none_at_all = [x for x in reclass if not x[2] and not x[3]]
    also_other = [x for x in reclass if x[2] and x[3]]
    print(f"- valid under assigned class: **{sum(1 for x in reclass if x[2])}**")
    print(f"- fails assigned class but valid under another: **{len(only_other)}**"
          + (f" → {', '.join(f'{i} ({g} → {o[0]})' for i,g,_,o in only_other)}"
             if only_other else ""))
    print(f"- valid under NO class: **{len(none_at_all)}**"
          + (f" → {', '.join(i for i,_,_,_ in none_at_all)}" if none_at_all else ""))
    print(f"- valid under assigned class AND others (ambiguous): "
          f"**{len(also_other)}**")

    # ---------------- per-model deltas ----------------
    print("\n## Per-model utilitarian accuracy, 1,535 vs 1,513\n")
    by_id = {it.id: it for it in insts}
    opt = {it.id: util_argmax_set(totals(it.cells)) for it in insts}
    choices = load_choices()
    print("| model | n (full) | acc 1,535 | n (kept) | acc 1,513 | delta (pp) |")
    print("|---|---|---|---|---|---|")
    deltas = []
    for m in sorted(choices):
        got = choices[m]
        full = [(i, cell) for i, cell in got.items() if i in by_id]
        kept = [(i, cell) for i, cell in full if i in keep_ids]
        a_full = sum(cell in opt[i] for i, cell in full) / len(full)
        a_keep = sum(cell in opt[i] for i, cell in kept) / len(kept)
        d = 100 * (a_keep - a_full)
        deltas.append(d)
        print(f"| {m} | {len(full)} | {a_full:.4f} | {len(kept)} | "
              f"{a_keep:.4f} | {d:+.3f} |")
    print(f"\n- mean delta **{sum(deltas)/len(deltas):+.3f} pp**, "
          f"max |delta| **{max(abs(x) for x in deltas):.3f} pp**, "
          f"range [{min(deltas):+.3f}, {max(deltas):+.3f}]")


if __name__ == "__main__":
    main()
