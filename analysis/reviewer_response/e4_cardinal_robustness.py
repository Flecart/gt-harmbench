"""Appendix E.4 extension (reviewer 1wTm): are the results artefacts of the
cardinal payoffs GPT-5.1 generated?

Computes, on the 1,535 release with the pinned baseline logs:
  1. joint payoff distribution: share of equal-payoff outcomes, correlation of
     the two players' payoffs at mutual (diagonal) and miscoordination outcomes;
  2. symmetry: cardinal (exact mirror) and ordinal-up-to-relabelling rates;
  3. ordinal determinacy: utilitarian argmax flip rate under monotone redraws,
     per game (0 for the four coordination-family games);
  4. utilitarian margin for PD and Chicken;
  5. robustness of model scores and ranking to 1,000 class-preserving monotone
     redraws, unrestricted and with PD constrained to keep 2R > T + S (the
     condition all 490 PD scenarios satisfy).

Writes data/e4_cardinal_robustness.json and figures/e4_*.png.
  PYTHONPATH=. uv run python -m analysis.reviewer_response.e4_cardinal_robustness
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.reviewer_response.lib import load_instances, load_choices, CELL_ORDER
from analysis.reviewer_response.ordinal_asymmetry_audit import sym_up_to_relabel

HERE = Path(__file__).resolve().parent
SEED, N = 20260727, 1000
GAMES = ["Prisoner's Dilemma", "Chicken", "Stag hunt", "Coordination", "Bach or Stravinski", "No conflict"]
PAPER = {"claude-4.5-opus": "Claude 4.5 Opus", "claude-4.5-sonnet": "Claude 4.5 Sonnet", "gpt-5.2": "GPT-5.2",
         "gpt-5.1": "GPT-5.1", "gpt-5-mini": "GPT-5 Mini", "gpt-5-nano": "GPT-5 Nano", "gpt-4o": "GPT-4o",
         "grok-4.1-fast": "Grok 4.1 Fast", "gemini-3-pro": "Gemini 3 Pro", "gemini-3-flash": "Gemini 3 Flash",
         "llama-3.3-70b": "Llama 3.3 70B", "llama-3.2-3b": "Llama 3.2 3B", "qwen3-30b": "Qwen3 30B",
         "qwen3-8b": "Qwen3 8B", "deepseek-v3.2": "DeepSeek V3.2"}
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"


def redraws(it, rng, constrain_pd):
    """(N,4) cell totals under monotone maps of the scenario's distinct payoff levels."""
    flat = sorted({v for c in it.cells for v in c})
    rank = {v: i for i, v in enumerate(flat)}
    cr = [(rank[a], rank[b]) for a, b in it.cells]
    out = np.empty((0, 4))
    while len(out) < N:
        U = np.sort(rng.uniform(-10, 10, size=(N, len(flat))), axis=1)
        T = np.stack([U[:, a] + U[:, b] for a, b in cr], axis=1)
        if constrain_pd and it.game == "Prisoner's Dilemma":
            diag = np.maximum(T[:, 0], T[:, 3]); off = np.maximum(T[:, 1], T[:, 2])
            T = T[diag > off]  # keep 2R > T + S
        out = np.vstack([out, T])
    return out[:N]


def main():
    insts = load_instances()
    ch = {PAPER[m]: v for m, v in load_choices().items() if m in PAPER}
    res = {}

    # 1. joint distribution
    diag_pairs = np.array([c for it in insts for k, c in enumerate(it.cells) if k in (0, 3)], float)
    off_pairs = np.array([c for it in insts for k, c in enumerate(it.cells) if k in (1, 2)], float)
    allp = np.vstack([diag_pairs, off_pairs])
    res["joint"] = {"frac_equal": float(np.mean(allp[:, 0] == allp[:, 1])),
                    "corr_mutual": float(np.corrcoef(diag_pairs.T)[0, 1]),
                    "corr_miscoord": float(np.corrcoef(off_pairs.T)[0, 1]),
                    "corr_all": float(np.corrcoef(allp.T)[0, 1])}
    # 2. symmetry
    card = [it.cells[0][0] == it.cells[0][1] and it.cells[3][0] == it.cells[3][1]
            and tuple(it.cells[1]) == tuple(reversed(it.cells[2])) for it in insts]
    res["symmetry"] = {"cardinal": float(np.mean(card)),
                       "ordinal_up_to_relabel": float(np.mean([sym_up_to_relabel(it.cells) for it in insts]))}
    # 4. margins
    marg = {}
    for g in ["Prisoner's Dilemma", "Chicken"]:
        m = []
        for it in insts:
            if it.game != g: continue
            t = it.totals; vals = [v for c in it.cells for v in c]
            m.append((max(t[0], t[3]) - max(t[1], t[2])) / (max(vals) - min(vals)) / 2 * 2)
        m = np.array(m); marg[g] = m
        res.setdefault("margin", {})[g] = {"mean": float(m.mean()), "frac_pos": float(np.mean(m > 0)),
                                         "frac_tie": float(np.mean(m == 0)), "frac_lt_0.10": float(np.mean(np.abs(m) < 0.10))}
    # 3 + 5. redraws
    for variant, cons in [("unrestricted", False), ("pd_constrained", True)]:
        rng = np.random.default_rng(SEED)
        flip = {g: [] for g in GAMES}
        hit = {m: np.zeros(N) for m in ch}; cnt = {m: 0 for m in ch}; base = {m: 0 for m in ch}
        for it in insts:
            T = redraws(it, rng, cons)
            opt = T >= T.max(axis=1, keepdims=True) - 1e-9           # (N,4) optimal cells per draw
            b = np.array(it.totals, float); bopt = b >= b.max()
            flip[it.game].append(float(np.mean(~(opt == bopt).all(axis=1))))
            for m, c in ch.items():
                k = c.get(it.id)
                if k is None: continue
                hit[m] += opt[:, k]; cnt[m] += 1; base[m] += bopt[k]
        rows = {m: {"unperturbed": base[m] / cnt[m], "mean": float((hit[m] / cnt[m]).mean()),
                    "p5": float(np.percentile(hit[m] / cnt[m], 5)), "p95": float(np.percentile(hit[m] / cnt[m], 95))}
                for m in ch}
        order0 = sorted(rows, key=lambda m: -rows[m]["unperturbed"]); order1 = sorted(rows, key=lambda m: -rows[m]["mean"])
        r0 = {m: i for i, m in enumerate(order0)}; r1 = {m: i for i, m in enumerate(order1)}
        x = np.array([r0[m] for m in rows]); y = np.array([r1[m] for m in rows])
        spear = float(1 - 6 * np.sum((x - y) ** 2) / (len(x) * (len(x) ** 2 - 1)))
        res[variant] = {"flip_rate": {g: float(np.mean(v)) for g, v in flip.items()}, "models": rows,
                        "spearman": spear, "max_rank_change": int(np.max(np.abs(x - y))),
                        "band_halfwidth_max": float(max((r["p95"] - r["p5"]) / 2 for r in rows.values())),
                        "shift_mean": float(np.mean([r["mean"] - r["unperturbed"] for r in rows.values()]))}
    (HERE / "data" / "e4_cardinal_robustness.json").write_text(json.dumps(res, indent=1))

    # figure: Chicken/PD margins + per-model robustness (PD-constrained)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    bins = np.linspace(-0.4, 1.0, 29)
    ax[0].hist(marg["Prisoner's Dilemma"], bins=bins, color=BLUE, alpha=0.85, label="Prisoner's Dilemma")
    ax[0].hist(marg["Chicken"], bins=bins, color=ORANGE, alpha=0.85, label="Chicken")
    ax[0].axvline(0, color=INK2, ls="--", lw=1)
    ax[0].set_xlabel("Normalised utilitarian margin of mutual cooperation"); ax[0].set_ylabel("Scenarios")
    ax[0].legend(frameon=False, fontsize=9); ax[0].set_title("Margin to the flip point", fontsize=10.5, loc="left")
    R = res["pd_constrained"]["models"]; order = sorted(R, key=lambda m: R[m]["unperturbed"])
    y = np.arange(len(order))
    ax[1].scatter([R[m]["unperturbed"] for m in order], y, color=INK2, s=22, zorder=3, label="Reported")
    ax[1].errorbar([R[m]["mean"] for m in order], y, xerr=[[R[m]["mean"] - R[m]["p5"] for m in order],
                   [R[m]["p95"] - R[m]["mean"] for m in order]], fmt="o", color=BLUE, ms=4, capsize=2,
                   label="1,000 redraws (mean, 5–95%)", zorder=3)
    ax[1].set_yticks(y); ax[1].set_yticklabels(order, fontsize=8.5); ax[1].set_xlim(0.2, 1.0)
    ax[1].set_xlabel("Utilitarian accuracy"); ax[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    ax[1].set_title("Scores under ordinal-preserving redraws", fontsize=10.5, loc="left")
    for a in ax:
        a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
        for s in ["top", "right"]: a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "e4_cardinal_robustness.png", dpi=200); plt.close(fig)
    print(json.dumps({k: v for k, v in res.items() if k not in ("unrestricted", "pd_constrained")}, indent=1))
    for v in ("unrestricted", "pd_constrained"):
        r = res[v]; print(v, "flip:", {g[:5]: round(x, 3) for g, x in r["flip_rate"].items()},
                          "spearman", round(r["spearman"], 3), "max rank change", r["max_rank_change"],
                          "band ±", round(r["band_halfwidth_max"], 3), "mean shift", round(r["shift_mean"], 3))


if __name__ == "__main__":
    main()
