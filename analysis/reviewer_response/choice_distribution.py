"""Appendix choice-distribution figures (distribution-0.png, distribution-2.png).

Per model x game: share of scenarios ending in each joint profile, over parsed
answers (n in the corner), on the 1,535 release. Actions are re-oriented per
scenario from its payoffs so each quadrant means the same thing in every game
instance (orient()): PD cooperate/defect, Chicken swerve/straight, Stag Hunt
stag/hare, BoS insist/concede, No Conflict best/other; Coordination keeps the
listed order (its two coordination points are equally good). Same pinned logs and cache as
Table 2. Style follows analysis/plots/shared.render_action_probability_grid.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.choice_distribution
"""
from __future__ import annotations

import ast, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.reviewer_response.reproduce_tables import CLEAN, CACHE
from analysis.reviewer_response.emit_tables import FINAL

HERE = Path(__file__).resolve().parent
GAMES = ["Prisoner's Dilemma", "Chicken", "Bach or Stravinski", "Stag hunt", "Coordination", "No conflict"]


LABELS = {"Prisoner's Dilemma": ("Cooperate", "Defect"), "Chicken": ("Swerve", "Straight"),
          "Bach or Stravinski": ("Insist", "Concede"), "Stag hunt": ("Stag", "Hare"),
          "Coordination": ("Option 1", "Option 2"), "No conflict": ("Best", "Other")}
CELLS = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]


def orient(r):
    """(row_map, col_map): listed action index -> canonical index (0 = first label)."""
    P = [[ast.literal_eval(r[CELLS[2 * i + j]]) for j in range(2)] for i in range(2)]
    g = r["formal_game"]
    tot = {(i, j): P[i][j][0] + P[i][j][1] for i in range(2) for j in range(2)}
    if g == "Prisoner's Dilemma":  # defect = dominant action for each player
        rd = 0 if all(P[0][j][0] >= P[1][j][0] for j in range(2)) else 1
        cd = 0 if all(P[i][0][1] >= P[i][1][1] for i in range(2)) else 1
        return {rd: 1, 1 - rd: 0}, {cd: 1, 1 - cd: 0}
    if g == "Chicken":  # straight = action in the crash (lowest-total) cell
        ri, ci = min(tot, key=tot.get)
        return {ri: 1, 1 - ri: 0}, {ci: 1, 1 - ci: 0}
    if g in ("Stag hunt", "No conflict"):  # stag / best = action in the highest-total cell
        ri, ci = max(tot, key=tot.get)
        return {ri: 0, 1 - ri: 1}, {ci: 0, 1 - ci: 1}
    if g == "Bach or Stravinski":  # insist = own favourite coordination point
        coord = sorted(tot, key=tot.get)[-2:]
        rbest = max(coord, key=lambda c: P[c[0]][c[1]][0]); cbest = max(coord, key=lambda c: P[c[0]][c[1]][1])
        return {rbest[0]: 0, 1 - rbest[0]: 1}, {cbest[1]: 0, 1 - cbest[1]: 1}
    return {0: 0, 1: 1}, {0: 0, 1: 1}


def render(models, mats, counts, path):
    fig = plt.figure(figsize=(len(GAMES) * 2.5, len(models) * 2.5))
    gs = fig.add_gridspec(len(models), len(GAMES), hspace=0.15, wspace=0.22)
    for i, m in enumerate(models):
        for j, g in enumerate(GAMES):
            ax = fig.add_subplot(gs[i, j])
            M = mats[m, g]
            ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=1)
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=9,
                            fontweight="bold", color="white" if M[r, c] > 0.5 else "black")
            # quadrant labels: rows = row player's 1st/2nd action, columns = column player's
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.tick_params(length=0, labelsize=7, colors="#52514e")
            a, b = LABELS[g]
            ax.set_xticklabels([a, b] if i == len(models) - 1 else [])
            ax.set_yticklabels([a, b], rotation=90, va="center")  # every panel: labels differ by game
            if i == 0:
                ax.set_title(g, fontsize=10, fontweight="bold", pad=10)
            if j == 0:
                ax.set_ylabel(m, fontsize=9, fontweight="bold", labelpad=4)
            ax.text(0.98, 0.02, f"n={counts[m, g]}", transform=ax.transAxes,
                    fontsize=7, ha="right", va="bottom", alpha=0.6)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = {r["id"]: r for r in csv.DictReader(open(CLEAN))}
    game = {i: r["formal_game"] for i, r in rows.items()}
    omap = {i: orient(r) for i, r in rows.items()}
    cache = json.loads(CACHE.read_text())
    mats, counts = {}, {}
    for m in FINAL:
        for g in GAMES:
            M = np.zeros((2, 2))
            for cid, (ri, ci) in cache[m].items():
                if game[cid] == g:
                    rm, cm = omap[cid]
                    M[rm[ri], cm[ci]] += 1
            counts[m, g] = int(M.sum())
            mats[m, g] = M / M.sum() if M.sum() else M
    models = list(FINAL)
    render(models[:8], mats, counts, HERE / "figures" / "distribution-0.png")
    render(models[8:], mats, counts, HERE / "figures" / "distribution-2.png")
    print("wrote figures/distribution-0.png (", ", ".join(models[:8]), ")\n      figures/distribution-2.png (",
          ", ".join(models[8:]), ")")


if __name__ == "__main__":
    main()
