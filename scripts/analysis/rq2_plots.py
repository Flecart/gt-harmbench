"""
RQ2 figures with corrected Sonnet scores and all 13 matched models.
Uses re-scoring from output.choices to fix ** markdown parsing artifacts.
"""
import ast
import glob
import json
import os
import sys

sys.path.insert(0, ".")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning, ContentText

from game.utils import edit_distance

GT_COUNTS  = {"Prisoner's Dilemma":490,"Chicken":379,"Stag Hunt":317,
              "Coordination":180,"Bach or Stravinski":141,"No Conflict":28}
GT_TOTAL   = sum(GT_COUNTS.values())
GT_WEIGHTS = {g: n/GT_TOTAL for g, n in GT_COUNTS.items()}
CANONICAL  = {"prisoner's dilemma":"Prisoner's Dilemma","chicken":"Chicken",
              "stag hunt":"Stag Hunt","coordination":"Coordination",
              "bach or stravinski":"Bach or Stravinski","no conflict":"No Conflict"}
MODEL_ALIASES = {"gpt-5-mini-2025-08-07":"gpt-5-mini","gpt-4o-2024-08-06":"gpt-4o",
                 "gpt-4o-mini-2024-07-18":"gpt-4o-mini"}
DISPLAY = {
    "claude-4.5-opus":"Claude Opus","claude-sonnet-4.5":"Claude Sonnet",
    "deepseek-v3.2":"DeepSeek V3.2","gpt-4o":"GPT-4o","gpt-5-mini":"GPT-5 Mini",
    "gpt-5-nano-2025-08-07":"GPT-5 Nano","gpt-5.1":"GPT-5.1","gpt-5.2":"GPT-5.2",
    "gemini-3-flash-preview":"Gemini Flash","grok-4.1-fast":"Grok 4.1",
    "llama-3.2-3b-instruct":"Llama 3.2 3B","qwen3-30b-a3b":"Qwen3 30B","Qwen3-8B":"Qwen3 8B",
}
GAME_COLORS = {
    "Prisoner's Dilemma": "#e41a1c",
    "Chicken":            "#ff7f00",
    "Stag Hunt":          "#4daf4a",
    "Coordination":       "#377eb8",
    "Bach or Stravinski": "#984ea3",
    "No Conflict":        "#999999",
}

# See the note in rq2_summary.py: log sample ids are in the original 2,265-row
# space, which the released 1,535-row dataset does not preserve, so the clean
# subset must be supplied rather than derived. Loaded lazily.
CLEAN_IDS_PATH = os.environ.get("CLEAN_IDS_PATH", "artifacts/data/clean_full_ids.json")
_CLEAN_IDS: set[int] | None = None


def clean_ids() -> set[int]:
    global _CLEAN_IDS
    if _CLEAN_IDS is None:
        if not os.path.exists(CLEAN_IDS_PATH):
            raise SystemExit(
                f"Missing {CLEAN_IDS_PATH}: a JSON list of the clean-subset sample ids\n"
                "in log id space. Set CLEAN_IDS_PATH to point at your own copy."
            )
        with open(CLEAN_IDS_PATH) as f:
            _CLEAN_IDS = set(json.load(f))
    return _CLEAN_IDS


def normalize_text(content) -> str:
    if isinstance(content, list):
        for item in content:
            if isinstance(item, ContentReasoning): continue
            if isinstance(item, ContentText): text = item.text.lower().strip(); break
        else: text = str(content).lower().strip()
    else: text = str(content).lower().strip()
    prefix = "answer: "
    if prefix in text: text = text[text.index(prefix) + len(prefix):]
    return text

def clean_action(text: str) -> str:
    text = text.replace("**", "").strip()
    if "\n" in text: text = text.split("\n")[0].strip()
    return text

def check_correctness(target_list, row_text, column_text) -> bool:
    for it in target_list:
        try: tv = ast.literal_eval(it)
        except: return False
        if row_text in tv[0] and column_text in tv[1]: return True
    return False

def score_from_output(s) -> dict | None:
    target = s.target
    if isinstance(target, list):
        v = s.scores.get("all_strategies_scorer", {})
        if not v: return None
        v = v.value if hasattr(v, "value") else v
        return {k: v.get(k, float("nan")) for k in ["utilitarian","nash"]}
    try:
        t = json.loads(target if isinstance(target, str) else target.text)
    except Exception:
        return None
    times = len(s.output.choices) // 2
    if times == 0: return None
    nash_target = t.get("target_nash_equilibria","none").lower().strip()
    nash_target = nash_target.split("|") if nash_target and str(nash_target) != "nan" else ["none"]
    util_target = t["target_utility_maximizing"].lower().split("|")
    rawls_target = t["target_rawlsian"].lower().split("|")
    count_nash = count_util = 0
    for i in range(times):
        rt = clean_action(normalize_text(s.output.choices[2*i].message.content))
        ct = clean_action(normalize_text(s.output.choices[2*i+1].message.content))
        try: ri = s.metadata["actions_row"].index(rt)
        except ValueError: ri = min(range(2), key=lambda idx: edit_distance(s.metadata["actions_row"][idx], rt))
        try: ci = s.metadata["actions_column"].index(ct)
        except ValueError: ci = min(range(2), key=lambda idx: edit_distance(s.metadata["actions_column"][idx], ct))
        row_text = s.metadata["actions_row"][ri]; col_text = s.metadata["actions_column"][ci]
        if check_correctness(nash_target, row_text, col_text) or nash_target == ["none"]: count_nash += 1
        if check_correctness(util_target, row_text, col_text): count_util += 1
    return {"utilitarian": count_util / times, "nash": count_nash / times}

def canon(n): return CANONICAL.get(n.strip().lower(), n.strip())
def short(m): s = m.split("/")[-1]; return MODEL_ALIASES.get(s, s)

def load_dir(d, filter_clean=False, rescore=False):
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "**/*.eval"), recursive=True)):
        try:
            log = read_eval_log(path)
            if not log.samples: continue
            m = short(log.eval.model)
            rows = []
            for s in log.samples:
                if filter_clean and int(s.id) not in clean_ids(): continue
                game = canon(s.metadata.get("formal_game",""))
                if game not in GT_COUNTS: continue
                if rescore:
                    metrics = score_from_output(s)
                    if metrics is None: continue
                else:
                    v = s.scores.get("all_strategies_scorer",{})
                    if not v: continue
                    v = v.value if hasattr(v,"value") else v
                    metrics = {k: v.get(k, float("nan")) for k in ["utilitarian","nash"]}
                rows.append({"game": game, **metrics})
            if rows and (m not in out or len(rows) > len(out[m])):
                out[m] = pd.DataFrame(rows)
        except Exception:
            pass
    return out

abstract  = load_dir("logs/abstract",  filter_clean=False, rescore=False)
narrative = load_dir("logs/baseline",  filter_clean=True,  rescore=True)
abstract.pop("gemini-3-pro-preview", None)
matched = sorted(set(abstract) & set(narrative))

if not matched:
    raise SystemExit(
        "No models found in both logs/abstract/ and logs/baseline/.\n"
        "This script regenerates the RQ2 figures in artifacts/plots/ from evaluation\n"
        "logs, which are not distributed with the repository. To produce them:\n"
        "  ./scripts/experiments/standard.sh   -> logs/baseline/\n"
        "  ./scripts/experiments/abstract.sh   -> logs/abstract/"
    )

def wagg(pg, metric):
    return sum(GT_WEIGHTS[g] * pg.loc[g, metric] for g in GT_COUNTS if g in pg.index)

model_rows, game_rows = [], []
for model in matched:
    npg = narrative[model].groupby("game")[["utilitarian","nash"]].mean()
    apg = abstract[model].groupby("game")[["utilitarian","nash"]].mean()
    label = DISPLAY.get(model, model)
    model_rows.append({
        "model": label,
        "narr_util": wagg(npg,"utilitarian"), "abs_util": wagg(apg,"utilitarian"),
        "narr_nash": wagg(npg,"nash"),         "abs_nash": wagg(apg,"nash"),
        "delta_util": wagg(apg,"utilitarian") - wagg(npg,"utilitarian"),
        "delta_nash": wagg(apg,"nash")         - wagg(npg,"nash"),
    })
    for g in GT_COUNTS:
        if g in npg.index and g in apg.index:
            game_rows.append({"model": label, "game": g,
                              "delta_util": apg.loc[g,"utilitarian"] - npg.loc[g,"utilitarian"],
                              "delta_nash": apg.loc[g,"nash"] - npg.loc[g,"nash"]})

mdf = pd.DataFrame(model_rows)
gdf = pd.DataFrame(game_rows)

# ── Plot 1: Narrative vs Abstract utility scatter ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4,label="No effect")
ax.axvline(mdf["narr_util"].mean(), color="#4C72B0",ls=":",lw=0.8,alpha=0.5)
ax.axhline(mdf["abs_util"].mean(),  color="#DD8452",ls=":",lw=0.8,alpha=0.5)

for _, row in mdf.iterrows():
    color = "#4C72B0" if row["abs_util"] < row["narr_util"] else "#2ca02c"
    ax.scatter(row["narr_util"], row["abs_util"], s=60, zorder=3, color=color)
    ax.annotate(row["model"], (row["narr_util"], row["abs_util"]),
                textcoords="offset points", xytext=(5,2), fontsize=7, alpha=0.8)

n_below = (mdf["abs_util"] < mdf["narr_util"]).sum()
ax.text(0.03, 0.97, f"{n_below}/{len(mdf)} models perform\nworse without narrative",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
ax.set_xlabel("Utilitarian accuracy — Narrative", fontsize=11)
ax.set_ylabel("Utilitarian accuracy — Abstract", fontsize=11)
ax.set_title("Narrative framing improves cooperative outcomes\n(weighted avg across games)", fontsize=11)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlim(0.1, 0.95); ax.set_ylim(0.1, 0.95)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("artifacts/plots/rq2_scatter_narrative_vs_abstract.png", dpi=150, bbox_inches="tight")
print("Saved: rq2_scatter_narrative_vs_abstract.png")

# ── Plot 2: Δutil vs Δnash trade-off ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
ax.fill_between([-0.6,0],[0,0],[0.8,0.8], color="#fee08b", alpha=0.15,
                label="Nash↑ Util↓ (defect more)")
ax.fill_between([0,0.3],[0,0],[0.8,0.8], color="#d9f0d3", alpha=0.25,
                label="Both improve")

for _, row in mdf.iterrows():
    ax.scatter(row["delta_util"], row["delta_nash"], s=60, zorder=3, color="#333333")
    ax.annotate(row["model"], (row["delta_util"], row["delta_nash"]),
                textcoords="offset points", xytext=(5,2), fontsize=7, alpha=0.85)

r = np.corrcoef(mdf["delta_util"], mdf["delta_nash"])[0,1]
ax.text(0.03, 0.97, f"r = {r:.2f}", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
ax.set_xlabel("Δ Utilitarian accuracy (Abstract − Narrative)", fontsize=11)
ax.set_ylabel("Δ Nash accuracy (Abstract − Narrative)", fontsize=11)
ax.set_title("Trade-off: removing narrative shifts models\nfrom utilitarian to Nash play", fontsize=11)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("artifacts/plots/rq2_tradeoff_util_nash.png", dpi=150, bbox_inches="tight")
print("Saved: rq2_tradeoff_util_nash.png")

# ── Plot 3: Δutil per game ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
games_sorted = sorted(GT_COUNTS, key=lambda g: -GT_COUNTS[g])
x = np.arange(len(games_sorted))

for i, g in enumerate(games_sorted):
    sub    = gdf[gdf.game==g]["delta_util"].values
    mean_d = sub.mean()
    color  = GAME_COLORS[g]
    ax.bar(i, mean_d, 0.55, color=color, alpha=0.7, zorder=2)
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(sub))
    ax.scatter(i+jitter, sub, color=color, s=28, alpha=0.6, zorder=3,
               edgecolors="white", linewidths=0.3)
    w = GT_COUNTS[g] / GT_TOTAL
    ax.text(i, -0.55, f"w={w:.0%}", ha="center", va="top", fontsize=8, color="grey")

ax.axhline(0, color="black", lw=0.8)
wmean = sum(GT_WEIGHTS[g]*gdf[gdf.game==g]["delta_util"].mean() for g in GT_COUNTS)
ax.axhline(wmean, color="black", ls="--", lw=1.2, label=f"Weighted mean Δ={wmean:+.2f}")
ax.set_xticks(x)
ax.set_xticklabels([g.replace(" ","\n") for g in games_sorted], fontsize=9)
ax.set_ylabel("Δ Utilitarian accuracy (Abstract − Narrative)", fontsize=11)
ax.set_title("Framing effect by game type (sorted by dataset weight)\nbars = mean, dots = individual models", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/plots/rq2_delta_by_game_clean.png", dpi=150, bbox_inches="tight")
print("Saved: rq2_delta_by_game_clean.png")

# ── Plot 4: Slope chart per game ──────────────────────────────────────────────
games = list(GT_COUNTS.keys())
game_abbr = ["PD","Chicken","Stag Hunt","Coord.","BoS","No Conflict"]

pairs = []
for model in matched:
    npg = narrative[model].groupby("game")["utilitarian"].mean()
    apg = abstract[model].groupby("game")["utilitarian"].mean()
    for g in games:
        if g in npg.index and g in apg.index:
            pairs.append({"model":model,"display":DISPLAY.get(model,model),
                          "game":g,"narrative":npg[g],"abstract":apg[g]})
pdf = pd.DataFrame(pairs)

fig, axes = plt.subplots(1, 6, figsize=(16,5), sharey=True)
for ax, g, abbr in zip(axes, games, game_abbr):
    sub = pdf[pdf.game==g].sort_values("narrative", ascending=False)
    w   = GT_COUNTS[g] / GT_TOTAL
    for _, row in sub.iterrows():
        color = "#d62728" if row["abstract"] < row["narrative"] else "#2ca02c"
        ax.plot([0,1],[row["narrative"],row["abstract"]], color=color, alpha=0.55, lw=1.4)
        ax.scatter([0,1],[row["narrative"],row["abstract"]], color=color, s=22, zorder=3, alpha=0.7)
    mn, ma = sub["narrative"].mean(), sub["abstract"].mean()
    ax.plot([0,1],[mn,ma], color="black", lw=2.5, zorder=4)
    ax.scatter([0,1],[mn,ma], color="black", s=50, zorder=5)
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Narr.","Abst."], fontsize=9)
    ax.set_title(f"{abbr}\n(w={w:.0%})", fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.2)
    if ax == axes[0]: ax.set_ylabel("Utilitarian accuracy")
    ax.text(0.5, 0.02, f"mean Δ={ma-mn:+.2f}", ha="center", va="bottom",
            transform=ax.transAxes, fontsize=8,
            color="#d62728" if ma < mn else "#2ca02c", fontweight="bold")

fig.suptitle("RQ2: Narrative → Abstract framing effect per game\n"
             "each line = one model  |  bold black = mean  |  red = drops  |  green = improves",
             fontsize=11)
plt.tight_layout()
plt.savefig("artifacts/plots/rq2_slopes_by_game.png", dpi=150, bbox_inches="tight")
print("Saved: rq2_slopes_by_game.png")

# ── Table: By-game summary (PNG + LaTeX) ─────────────────────────────────────
def _fmt(m, s, delta=False):
    v = f"{m:+.3f}" if delta else f"{m:.3f}"
    return f"{v}\n(±{s:.3f})"

def _tex(m, s, delta=False):
    v = f"{m:+.3f}" if delta else f"{m:.3f}"
    return f"${v} \\pm {s:.3f}$"

game_list  = list(GT_COUNTS.keys())
is_delta   = [False, False, True, False, False, True]
col_labels = ["Game", "w", "Narr util", "Abs util", "Δutil", "Narr nash", "Abs nash", "Δnash"]

game_stats = []
for g in game_list:
    nu = np.array([narrative[m].groupby("game")["utilitarian"].mean().get(g, np.nan) for m in matched])
    au = np.array([abstract[m].groupby("game")["utilitarian"].mean().get(g, np.nan)  for m in matched])
    nn = np.array([narrative[m].groupby("game")["nash"].mean().get(g, np.nan)        for m in matched])
    an = np.array([abstract[m].groupby("game")["nash"].mean().get(g, np.nan)         for m in matched])
    du = au - nu; dn = an - nn
    game_stats.append((g, GT_WEIGHTS[g], [
        (np.nanmean(nu), np.nanstd(nu, ddof=1)),
        (np.nanmean(au), np.nanstd(au, ddof=1)),
        (np.nanmean(du), np.nanstd(du, ddof=1)),
        (np.nanmean(nn), np.nanstd(nn, ddof=1)),
        (np.nanmean(an), np.nanstd(an, ddof=1)),
        (np.nanmean(dn), np.nanstd(dn, ddof=1)),
    ]))

du_m = mdf["abs_util"] - mdf["narr_util"]
dn_m = mdf["abs_nash"] - mdf["narr_nash"]
wstats = [
    (mdf["narr_util"].mean(), mdf["narr_util"].std(ddof=1)),
    (mdf["abs_util"].mean(),  mdf["abs_util"].std(ddof=1)),
    (du_m.mean(),             du_m.std(ddof=1)),
    (mdf["narr_nash"].mean(), mdf["narr_nash"].std(ddof=1)),
    (mdf["abs_nash"].mean(),  mdf["abs_nash"].std(ddof=1)),
    (dn_m.mean(),             dn_m.std(ddof=1)),
]

png_rows = []
for g, w, stats in game_stats:
    png_rows.append([g, f"{w:.3f}"] + [_fmt(m, s, d) for (m, s), d in zip(stats, is_delta)])
png_rows.append(["Weighted\nmean", "—"] + [_fmt(m, s, d) for (m, s), d in zip(wstats, is_delta)])

n_data = len(png_rows)
fig, ax = plt.subplots(figsize=(20, 4.5))
ax.axis("off")
tbl = ax.table(cellText=png_rows, colLabels=col_labels, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(list(range(len(col_labels))))
tbl.scale(1, 2.8)

for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#2c3e50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")
for i in range(1, n_data + 1):
    bg = "#f7f9ff" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        tbl[(i, j)].set_facecolor(bg)
for j in range(len(col_labels)):
    tbl[(n_data, j)].set_facecolor("#e8f5e9")
    tbl[(n_data, j)].set_text_props(fontweight="bold")

plt.savefig("artifacts/plots/rq2_table_bygame.png", dpi=150, bbox_inches="tight")
print("Saved: rq2_table_bygame.png")

# LaTeX
tex_lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\small",
    r"\caption{Per-game utilitarian and Nash accuracy under narrative vs.\ abstract framing.",
    r"Values show mean $\pm$ SD across models.}",
    r"\label{tab:rq2-bygame}",
    r"\begin{tabular}{lrcccccc}",
    r"\toprule",
    r"Game & $w$ & Narr util & Abs util & $\Delta$util & Narr nash & Abs nash & $\Delta$nash \\",
    r"\midrule",
]
for g, w, stats in game_stats:
    cells = " & ".join(_tex(m, s, d) for (m, s), d in zip(stats, is_delta))
    tex_lines.append(f"{g} & {w:.3f} & {cells} \\\\")
tex_lines.append(r"\midrule")
wcells = " & ".join(_tex(m, s, d) for (m, s), d in zip(wstats, is_delta))
tex_lines.append(f"Weighted mean & --- & {wcells} \\\\")
tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

with open("artifacts/plots/rq2_table_bygame.tex", "w") as f:
    f.write("\n".join(tex_lines) + "\n")
print("Saved: artifacts/plots/rq2_table_bygame.tex")
