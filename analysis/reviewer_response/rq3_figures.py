"""RQ3 paper figures from the Sonnet 5.5 judge labels (camera-ready rerun):
Figure 5 (reasoning.png: category frequency by outcome) and the appendix heatmaps
(reasoning-app1.png by game, reasoning-app2.png by model). Same style as the
submitted figures.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.rq3_figures
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = HERE / "data/judge_full/sonnet55_release_labels.jsonl"
FIG = HERE / "figures"
CATS = ["NASH_DOMINANT", "PAYOFF_MAX", "UTILITARIAN", "RAWLSIAN", "CATASTROPHE_PREVENTION",
        "PRECAUTIONARY", "AI_SAFETY", "DOMAIN_OTHER"]
BAR_NAME = {"PAYOFF_MAX": "Payoff Max.", "NASH_DOMINANT": "Nash/Dominant", "PRECAUTIONARY": "Precautionary",
            "CATASTROPHE_PREVENTION": "Catastophe Pre.", "DOMAIN_OTHER": "Domain-Specific",
            "UTILITARIAN": "Utilitarian", "AI_SAFETY": "AI Safety", "RAWLSIAN": "Rawlsian"}
BAR_NAME["CATASTROPHE_PREVENTION"] = "Catastrophe Pre."
HM_NAME = {"NASH_DOMINANT": "Nash/Dom.", "PAYOFF_MAX": "Payoff Max", "UTILITARIAN": "Utilitarian",
           "RAWLSIAN": "Rawlsian", "CATASTROPHE_PREVENTION": "Catastrophe", "PRECAUTIONARY": "Precaution.",
           "AI_SAFETY": "AI Safety", "DOMAIN_OTHER": "Domain"}
GAMES = [("Bach or Stravinski", "BoS"), ("Chicken", "Chicken"), ("Coordination", "Coord."),
         ("No conflict", "No Confl."), ("Prisoner's Dilemma", "PD"), ("Stag hunt", "Stag")]
MODELS = [("anthropic/claude-opus-4.5", "Opus 4.5"), ("anthropic/claude-sonnet-4.5", "Sonnet 4.5"),
          ("deepseek/deepseek-v3.2", "DeepSeek"), ("qwen/qwen3-30b-a3b", "Qwen3")]


def freq(rows):
    return {c: float(np.mean([r["labels"][c] for r in rows])) for c in CATS}


def fig_outcome(R):
    opt = freq([r for r in R if r["util_optimal"] == 1]); sub = freq([r for r in R if r["util_optimal"] == 0])
    order = sorted(CATS, key=lambda c: opt[c] - sub[c])  # most negative gap on top
    fig, ax = plt.subplots(figsize=(10.5, 6.5)); ax.set_facecolor("#f0eff4")
    y = np.arange(len(order))[::-1]; h = 0.35
    ax.barh(y + h / 2, [sub[c] for c in order], h, color="#dc5a5a", label="Suboptimal Outcome", zorder=3)
    ax.barh(y - h / 2, [opt[c] for c in order], h, color="#5fa5d9", label="Optimal Outcome", zorder=3)
    for yi, c in zip(y, order):
        ax.text(sub[c] + 0.01, yi + h / 2, f"{sub[c]:.2f}", va="center", fontsize=11, fontweight="bold", color="#d62728")
        ax.text(opt[c] + 0.01, yi - h / 2, f"{opt[c]:.2f}", va="center", fontsize=11, fontweight="bold", color="#5fa5d9")
    ax.set_yticks(y); ax.set_yticklabels([BAR_NAME[c] for c in order], fontsize=14)
    ax.set_xlim(0, 1); ax.set_xlabel("Frequency", fontsize=14); ax.tick_params(axis="x", labelsize=12)
    ax.xaxis.grid(True, ls="--", color="#d0d0d0", zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    h_, l_ = ax.get_legend_handles_labels(); ax.legend(h_[::-1], l_[::-1], loc="lower right", fontsize=14)
    ax.set_title("Frequency of Reasoning Categories by Outcome Type", fontsize=18, fontweight="bold", pad=14)
    fig.tight_layout(); fig.savefig(FIG / "reasoning.png", dpi=300); plt.close(fig)
    return opt, sub


def heatmap(M, cols, title, cmap, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if M[i, j] > 0.45 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels([HM_NAME[c] for c in CATS], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def reparsed_optimal(R):
    """Replace the logs' stored utilitarian_score (wrong for Sonnet 4.5: its
    '**ANSWER:**' format fails the scorer's exact match) with the joint outcome of
    the re-parsed choices behind Table 2 (reproduce_tables cache); an unparsed
    answer counts as not optimal, as in Table 2."""
    import csv
    from analysis.reviewer_response import judge_pilot as jp
    from analysis.reviewer_response.reproduce_tables import CLEAN, CACHE, target_cells
    rows = list(csv.DictReader(open(CLEAN)))
    by_story = {}
    for r in rows:
        by_story[r["story_row"].strip()] = r; by_story[r["story_col"].strip()] = r
    story = {(t["id"], t["model"]): t["story"].strip() for t in jp.load_traces()}
    label = {"anthropic/claude-opus-4.5": "Claude 4.5 Opus", "anthropic/claude-sonnet-4.5": "Claude 4.5 Sonnet",
             "deepseek/deepseek-v3.2": "DeepSeek V3.2", "qwen/qwen3-30b-a3b": "Qwen3 30B"}
    cache = json.loads(CACHE.read_text())
    for r in R:
        row = by_story[story[(r["id"], r["model_judged"])]]
        ch = cache[label[r["model_judged"]]].get(row["id"])
        r["util_optimal_stored"] = r["util_optimal"]
        r["util_optimal"] = int(ch is not None and tuple(ch) in target_cells(row, "target_utility_maximizing"))
    return R


def main():
    R = reparsed_optimal([json.loads(l) for l in open(LABELS) if '"labels": {' in l])
    share = defaultdict(list)
    for r in R: share[r["model_judged"]].append((r["util_optimal_stored"], r["util_optimal"]))
    for m, v in share.items():
        print(f"  {m:30s} optimal share stored {np.mean([a for a, _ in v]):.3f} -> re-parsed {np.mean([b for _, b in v]):.3f}")
    opt, sub = fig_outcome(R)
    byg = defaultdict(list); bym = defaultdict(list)
    for r in R: byg[r["game"]].append(r); bym[r["model_judged"]].append(r)
    G = np.array([[freq(byg[g])[c] for g, _ in GAMES] for c in CATS])
    Mm = np.array([[freq(bym[m])[c] for m, _ in MODELS] for c in CATS])
    heatmap(G, [s for _, s in GAMES], "Category frequency by game type", "YlOrRd", FIG / "reasoning-app1.png")
    heatmap(Mm, [s for _, s in MODELS], "Category frequency by model", "YlGnBu", FIG / "reasoning-app2.png")
    out = {"n": len(R), "optimal": opt, "suboptimal": sub, "delta": {c: opt[c] - sub[c] for c in CATS},
           "by_game": {s: dict(zip(CATS, G[:, j])) for j, (_, s) in enumerate(GAMES)},
           "by_model": {s: dict(zip(CATS, Mm[:, j])) for j, (_, s) in enumerate(MODELS)}}
    (HERE / "data/rq3_judge_summary.json").write_text(json.dumps(out, indent=1))
    for c in CATS:
        gi = int(np.argmax(G[CATS.index(c)])); mi = int(np.argmax(Mm[CATS.index(c)]))
        print(f"{c:24s} Δ {opt[c]-sub[c]:+.3f} | highest game {GAMES[gi][1]} {G[CATS.index(c), gi]:.2f} | highest model {MODELS[mi][1]} {Mm[CATS.index(c), mi]:.2f}")


if __name__ == "__main__":
    main()
