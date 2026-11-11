"""Camera-ready RQ4 figures from the merged mechanism results (corrected
recommendation, 1,535 release, re-parsed answers, unparseable = wrong).

Writes the six images the paper includes, same filenames and visual style:
  MD-Figure6.png  MD-Figure7.png  fig4a_variant_util.png  fig4b_variant_nash.png
  md_heatmap_games.png  md_heatmap_models.png
plus rq4_numbers.json with every number quoted in the text.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.rq4_figures [--out DIR]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from analysis.reviewer_response.mech_rerun_merge import (
    OLD_CACHE, NEW_CACHE, NEW_MODELS, CONDITIONS, MECHS, VARIANTS, old_key, meta)
from analysis.reviewer_response.mechanism_compliance import load_universe, pick

OUT = Path(__file__).resolve().parent / "figures" / "rq4"
MODEL_LABEL = {"gpt-5-mini": "GPT-5 Mini", "deepseek-v3.2": "DeepSeek V3.2", "claude-4.5-sonnet": "Claude 4.5",
               "llama-3.2-3b": "Llama 3.2 3B", "qwen-3-30b": "Qwen3 30B", "gemini-3-flash-preview": "Gemini 3 Flash"}
MODELS = ["deepseek-v3.2", "claude-4.5-sonnet", "llama-3.2-3b", "qwen-3-30b", "gemini-3-flash-preview", "gpt-5-mini"]
MECH_LABEL = {"message": "Messages", "contracts": "Contracts", "payments": "Payments",
              "penalties": "Penalties", "mediator": "Mediator"}
MECH_ORDER = ["message", "contracts", "payments", "penalties", "mediator"]
VAR_LABEL = {"v1_initial": "Initial", "v2_formal": "Formal", "v3_credibility": "Credibility", "v4_moral": "Moral"}
VAR_ROW = {"v1_initial": "Initial", "v2_formal": "Formal", "v3_credibility": "Credible", "v4_moral": "Moral"}
VAR_COLOR = {"Initial": "#1f77b4", "Formal": "#ff7f0e", "Credibility": "#2ca02c", "Moral": "#d62728"}
GAMES = [("Prisoner's Dilemma", "PD"), ("Stag hunt", "Stag Hunt"), ("Chicken", "Chicken"),
         ("Bach or Stravinski", "BoS"), ("Coordination", "Coord."), ("No conflict", "No Confl.")]
NASH_C, UTIL_C, UP_C, DOWN_C = "#9ec3fb", "#ffcc99", "#a8e4a0", "#ff9999"
UP_T, DOWN_T = "#00b050", "#a52a2a"


def scores():
    """{(model, cond): {'util': [0/1 per scenario], 'nash': [...], 'game': [...]}}"""
    old = json.loads(OLD_CACHE.read_text()); new = json.loads(NEW_CACHE.read_text())
    _, released = load_universe(); M = meta(released)
    out = {}
    for m in MODELS:
        for c in CONDITIONS:
            d = dict(old[old_key(m, c)]); d.update(new.get(f"{m}/{c}", {}))
            u, n, g = [], [], []
            for i, mm in M.items():
                a = d.get(str(i)); cell = None
                if a:
                    r, cc = pick(a[0], mm["ar"]), pick(a[1], mm["ac"])
                    if r is not None and cc is not None:
                        cell = (r, cc)
                u.append(cell in mm["util"]); n.append(cell in mm["nash"]); g.append(mm["game"])
            out[(m, c)] = dict(util=np.array(u, float), nash=np.array(n, float), game=np.array(g))
    return out


def stacked(ax, labels, base_n, val_n, base_u, val_u, title, rot=0):
    x = np.arange(len(labels)); w = 0.35
    for xs, b, v, col in [(x - w / 2, base_n, val_n, NASH_C), (x + w / 2, base_u, val_u, UTIL_C)]:
        ax.bar(xs, b, w, color=col)
        d = np.array(v) - np.array(b)
        ax.bar(xs, d, w, bottom=b, color=[UP_C if e > 0.005 else DOWN_C if e < -0.005 else col for e in d])
        for xi, vi, di in zip(xs, v, d):
            top = max(vi, vi - di)
            if di < -0.005:  # decrease: label inside the bar, below the baseline
                ax.text(xi, vi - 0.06, f"{vi:.2f}", ha="center", fontsize=15, fontweight="bold")
                ax.text(xi, vi - 0.11, f"{di:+.2f}", ha="center", fontsize=15, fontweight="bold", color=DOWN_T)
            else:
                ax.text(xi, top + 0.012, f"{vi:.2f}", ha="center", fontsize=15, fontweight="bold")
                if di > 0.005:
                    ax.text(xi, top + 0.06, f"{di:+.2f}", ha="center", fontsize=15, fontweight="bold", color=UP_T)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=16, rotation=rot, ha="right" if rot else "center")
    ax.set_ylabel("Accuracy", fontsize=22, fontweight="bold")
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_ylim(0, 1.08); ax.set_facecolor("#eaeaf2"); ax.grid(axis="y", color="white"); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(OUT)); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = scores()
    acc = {k: (v["util"].mean(), v["nash"].mean()) for k, v in S.items()}
    mean = lambda f: float(np.mean([f(m) for m in MODELS]))
    base_u = mean(lambda m: acc[(m, "baseline")][0]); base_n = mean(lambda m: acc[(m, "baseline")][1])
    mech = {k: (mean(lambda m: np.mean([acc[(m, f"{k}-{v}")][0] for v in VARIANTS])),
                mean(lambda m: np.mean([acc[(m, f"{k}-{v}")][1] for v in VARIANTS]))) for k in MECH_ORDER}
    per_model = {m: dict(base=acc[(m, "baseline")],
                         mech=(np.mean([acc[(m, c)][0] for c in CONDITIONS[1:]]),
                               np.mean([acc[(m, c)][1] for c in CONDITIONS[1:]]))) for m in MODELS}

    # MD-Figure6: by mechanism
    fig, ax = plt.subplots(figsize=(16, 8.5))
    stacked(ax, [MECH_LABEL[k] for k in MECH_ORDER], [base_n] * 5, [mech[k][1] for k in MECH_ORDER],
            [base_u] * 5, [mech[k][0] for k in MECH_ORDER], "Mechanism Design: Impact on Game Outcomes")
    ax.axhline(base_n, color=NASH_C, ls="--", lw=1.5); ax.axhline(base_u, color=UTIL_C, ls="--", lw=1.5)
    ax.set_xlabel("Mechanism", fontsize=20, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=NASH_C, label="Nash Equilibrium"),
                       mpatches.Patch(color=UTIL_C, label="Utilitarian"),
                       plt.Line2D([0], [0], color=NASH_C, ls="--", label=f"Nash Baseline ({base_n:.2f})"),
                       plt.Line2D([0], [0], color=UTIL_C, ls="--", label=f"Util. Baseline ({base_u:.2f})")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=18, frameon=True, shadow=True)
    fig.tight_layout(); fig.savefig(out / "MD-Figure6.png", dpi=300, facecolor="white"); plt.close(fig)

    # MD-Figure7: by model
    fig, ax = plt.subplots(figsize=(16, 8.5))
    stacked(ax, [MODEL_LABEL[m] for m in MODELS], [per_model[m]["base"][1] for m in MODELS],
            [per_model[m]["mech"][1] for m in MODELS], [per_model[m]["base"][0] for m in MODELS],
            [per_model[m]["mech"][0] for m in MODELS],
            "Mechanism Design Impact by Model (Average Across All Mechanisms)", rot=20)
    ax.legend(handles=[mpatches.Patch(color=NASH_C, label="Nash Equilibrium"),
                       mpatches.Patch(color=UTIL_C, label="Utilitarian")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=18, frameon=True, shadow=True)
    fig.tight_layout(); fig.savefig(out / "MD-Figure7.png", dpi=300, facecolor="white"); plt.close(fig)

    # fig4a / fig4b: per variant, averaged over models
    for metric, idx, fname, base in [("Utilitarian Accuracy", 0, "fig4a_variant_util.png", base_u),
                                     ("Nash Accuracy", 1, "fig4b_variant_nash.png", base_n)]:
        fig, ax = plt.subplots(figsize=(20, 8.5)); bw, gap = 0.18, 0.3
        centers = []
        for i, k in enumerate(MECH_ORDER):
            s = i * (4 * bw + gap); centers.append(s + 1.5 * bw)
            for j, v in enumerate(VARIANTS):
                val = mean(lambda m: acc[(m, f"{k}-{v}")][idx]); d = val - base
                ax.bar(s + j * bw, val, bw, color=VAR_COLOR[VAR_LABEL[v]], edgecolor="black", lw=0.5)
                ax.text(s + j * bw, val + 0.01, f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
                if abs(d) > 0.005:
                    ax.text(s + j * bw, val + 0.045, f"{d:+.2f}", ha="center", fontsize=8, fontweight="bold",
                            color="#2ecc71" if d > 0 else "#e74c3c")
            if i:
                ax.axvline(s - gap / 2, color="gray", ls=":", alpha=0.5)
        ax.axhline(base, color="black", ls="--", alpha=0.7, lw=1.5)
        ax.set_xticks(centers); ax.set_xticklabels([MECH_LABEL[k] for k in MECH_ORDER], fontsize=13)
        ax.set_xlabel("Mechanism", fontsize=15, fontweight="bold"); ax.set_ylabel(metric, fontsize=15, fontweight="bold")
        ax.set_title(f"Prompt Variant Sensitivity ({metric})", fontsize=17, fontweight="bold")
        ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.3)
        ax.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in VAR_COLOR.items()]
                  + [plt.Line2D([0], [0], color="black", ls="--", label=f"Baseline ({base:.2f})")],
                  loc="upper left", ncol=3, fontsize=11, frameon=True, shadow=True)
        fig.tight_layout(); fig.savefig(out / fname, dpi=300, facecolor="white"); plt.close(fig)

    # heatmaps (utilitarian): rows = baseline + 20 variants
    rows = [("baseline", "Baseline")] + [(f"{k}-{v}", f"{MECH_LABEL[k].rstrip('s') if k == 'message' else MECH_LABEL[k]} {VAR_ROW[v]}")
                                         for k in ["message", "contracts", "payments", "penalties", "mediator"] for v in VARIANTS]
    games = np.array([[np.mean([S[(m, c)]["util"][S[(m, c)]["game"] == g].mean() for m in MODELS]) for g, _ in GAMES]
                      for c, _ in rows])
    models = np.array([[acc[(m, c)][0] for m in MODELS] for c, _ in rows])
    for mat, cols, title, xl, fname, size in [
            (games, [s for _, s in GAMES], "Mechanism Design × Game Type: Utilitarian Accuracy", "Game Type", "md_heatmap_games.png", (9, 10.5)),
            (models, [MODEL_LABEL[m] for m in MODELS], "Mechanism Design × Model: Utilitarian Accuracy", "Model", "md_heatmap_models.png", (10, 10.5))]:
        fig, ax = plt.subplots(figsize=size)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.3, vmax=0.9, aspect="auto")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if (v < 0.4 or v > 0.82) else "black")
        for y in [0.5, 4.5, 8.5, 12.5, 16.5]:
            ax.axhline(y, color="black", lw=1.5)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows], fontsize=10)
        ax.set_xlabel(xl, fontsize=13, fontweight="bold"); ax.set_ylabel("Mechanism Variant", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label("Utilitarian Accuracy", fontsize=12)
        fig.tight_layout(); fig.savefig(out / fname, dpi=300, facecolor="white"); plt.close(fig)

    nums = dict(baseline=dict(util=base_u, nash=base_n),
                mechanisms={k: dict(util=u, nash=n, d_util=u - base_u, d_nash=n - base_n,
                                    failure_reduction=(u - base_u) / (1 - base_u)) for k, (u, n) in mech.items()},
                models={m: dict(base_util=v["base"][0], base_nash=v["base"][1], mech_util=v["mech"][0],
                                mech_nash=v["mech"][1]) for m, v in per_model.items()},
                n_scenarios=1535, models_used=MODELS)
    (out / "rq4_numbers.json").write_text(json.dumps(nums, indent=1, default=float))
    print(json.dumps(nums, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
