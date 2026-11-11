"""Cross-play from existing self-play logs (no new API calls).

The two role queries are independent and contain no opponent information (see
eval.py TwoWaySolver), so model A's row choice and model B's column choice are
each fixed regardless of partner. Cross-play (A as row, B as column) is therefore
the stored profile (A.row_idx, B.col_idx), scored against the scenario payoffs.
All n^2 pairings reconstruct from the n self-play runs.

Read-only. Metric: utilitarian accuracy (joint profile is a utilitarian argmax),
on the clean 1535. Choices come from the same pinned logs and cache as Table 2
(reproduce_tables.LOGS / baseline_choices.json). Each pairing is scored on the
scenarios where both models' answers parsed. Counting unparsed answers as wrong
(as Table 2 does) would penalise a pairing for either side's parse failures but
self-play only for one, biasing cross-play down (~1.7 pp); so the diagonal here
is lenient self-play, slightly above Table 2.
Usage: PYTHONPATH=. uv run python -m analysis.reviewer_response.cross_play
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.reviewer_response.reproduce_tables import CLEAN, CACHE, target_cells
from analysis.reviewer_response.emit_tables import FINAL

HERE = Path(__file__).resolve().parent
CLASSES = ["Prisoner's Dilemma", "Chicken", "Stag hunt", "Coordination",
           "Bach or Stravinski", "No conflict"]


def main():
    rows = {r["id"]: r for r in csv.DictReader(open(CLEAN))}
    game = {i: r["formal_game"] for i, r in rows.items()}
    optimal = {i: target_cells(r, "target_utility_maximizing") for i, r in rows.items()}
    cache = json.loads(CACHE.read_text())

    # per model: {instance: (row_idx, col_idx)}, paper models in table order
    models = list(FINAL)
    row_of = {m: {i: rc[0] for i, rc in cache[m].items()} for m in models}
    col_of = {m: {i: rc[1] for i, rc in cache[m].items()} for m in models}

    def util_ok(iid, ri, ci):
        return (ri, ci) in optimal[iid]

    # full grid + per-game breakdown
    grid = np.full((len(models), len(models)), np.nan)
    self_vals, cross_vals = [], []
    per_game = {g: {"self": [], "cross": []} for g in CLASSES}
    for a, A in enumerate(models):
        for b, B in enumerate(models):
            common = [i for i in rows if i in row_of[A] and i in col_of[B]]
            hits = [util_ok(i, row_of[A][i], col_of[B][i]) for i in common]
            acc = float(np.mean(hits))
            grid[a, b] = acc
            (self_vals if a == b else cross_vals).append(acc)
            for i, h in zip(common, hits):
                per_game[game[i]]["self" if a == b else "cross"].append(h)

    # validation: diagonal should equal self-play utilitarian accuracy
    print("=== self-play (diagonal) vs cross-play (off-diagonal) ===")
    print(f"self-play  mean utilitarian accuracy: {np.mean(self_vals):.3f}")
    print(f"cross-play mean utilitarian accuracy: {np.mean(cross_vals):.3f}")
    print(f"difference (cross - self): {np.mean(cross_vals)-np.mean(self_vals):+.3f}")
    print(f"cross-play spread: min {np.min(cross_vals):.3f} max {np.max(cross_vals):.3f} "
          f"sd {np.std(cross_vals):.3f}")

    print("\n=== per game type: self vs cross (utilitarian accuracy) ===")
    print(f"{'game':22s} {'self':>6} {'cross':>6} {'diff':>7}")
    for g in CLASSES:
        s = np.mean(per_game[g]["self"]); c = np.mean(per_game[g]["cross"])
        print(f"{g:22s} {s:6.3f} {c:6.3f} {c-s:+7.3f}")

    # best/worst partners on average (column player B averaged over A)
    col_mean = np.nanmean(grid, axis=0)
    row_mean = np.nanmean(grid, axis=1)
    print("\n=== as COLUMN partner: mean accuracy induced (best/worst) ===")
    order = np.argsort(-col_mean)
    for k in list(order[:3]) + list(order[-3:]):
        print(f"  {models[k]:20s} {col_mean[k]:.3f}")

    # write grid + figure
    with open(HERE / "data" / "cross_play_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_model\\col_model"] + models)
        for a, A in enumerate(models):
            w.writerow([A] + [f"{grid[a,b]:.4f}" if not np.isnan(grid[a,b]) else ""
                              for b in range(len(models))])

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(grid, cmap="viridis", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=90, fontsize=7)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=7)
    ax.set_xlabel("column player"); ax.set_ylabel("row player")
    ax.set_title("Cross-play utilitarian accuracy (diagonal = self-play)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "cross_play_grid.png", dpi=130)
    plt.close(fig)

    # paper figure (appendix fig:cross-play), same style as the submitted one
    fig, ax = plt.subplots(figsize=(15, 12.5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
    for a in range(len(models)):
        for b in range(len(models)):
            ax.text(b, a, f"{grid[a, b]:.2f}", ha="center", va="center", fontsize=10,
                    fontweight="bold" if a == b else "normal")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=10)
    ax.set_xticks(np.arange(-.5, len(models)), minor=True)
    ax.set_yticks(np.arange(-.5, len(models)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Column player", fontsize=13); ax.set_ylabel("Row player", fontsize=13)
    ax.set_title("Utilitarian accuracy: row model \u00d7 column model (1,535 scenarios; diagonal = self-play)",
                 fontsize=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03); cb.set_label("Utilitarian accuracy", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "cross_play_utilitarian.png", dpi=150)
    plt.close(fig)
    print("\nwrote cross_play_grid.csv + figures/cross_play_grid.png + figures/cross_play_utilitarian.png")
    partner_summary(rows, optimal, row_of, col_of, models)


def partner_summary(rows, optimal, row_of, col_of, models):
    """Per model: self-play vs mean cross-play (both roles, 14 partners), overall and per game.
    Writes data/cross_play_summary.json and the LaTeX rows for the appendix table."""
    def acc(A, B, game=None):
        ids = [i for i in rows if i in row_of[A] and i in col_of[B] and (game is None or rows[i]["formal_game"] == game)]
        return float(np.mean([(row_of[A][i], col_of[B][i]) in optimal[i] for i in ids]))
    games = ["Prisoner's Dilemma", "Chicken", "Bach or Stravinski", "Stag hunt", "Coordination", "No conflict"]
    out = {}
    for A in models:
        others = [B for B in models if B != A]
        cross = lambda g=None: float(np.mean([(acc(A, B, g) + acc(B, A, g)) / 2 for B in others]))
        part = {B: (acc(A, B) + acc(B, A)) / 2 for B in others}
        out[A] = {"self": acc(A, A), "cross": cross(),
                  "delta_game": {g: cross(g) - acc(A, A, g) for g in games},
                  "best_partner": max(part, key=part.get), "worst_partner": min(part, key=part.get)}
    s_ = [out[m]["self"] for m in models]; d_ = [out[m]["cross"] - out[m]["self"] for m in models]
    out["_corr_self_delta"] = float(np.corrcoef(s_, d_)[0, 1])
    (HERE / "data" / "cross_play_summary.json").write_text(json.dumps(out, indent=1))
    lines = []
    for m in models:
        o = out[m]; d = 100 * (o["cross"] - o["self"])
        lines.append(f"{m} & {o['self']:.2f} & {o['cross']:.2f} & {d:+.1f} & "
                     + " & ".join(f"{100 * o['delta_game'][g]:+.1f}" for g in games) + r" \\")
    md = {g: 100 * np.mean([out[m]["delta_game"][g] for m in models]) for g in games}
    lines.append(r"\midrule")
    lines.append(r"\textbf{Mean} & & & " + f"{100 * np.mean(d_):+.1f} & " + " & ".join(f"{md[g]:+.1f}" for g in games) + r" \\")
    (HERE / "data" / "cross_play_summary_rows.tex").write_text("\n".join(lines) + "\n")
    print(f"partner summary: r(self, delta) = {out['_corr_self_delta']:.2f}")


if __name__ == "__main__":
    main()
