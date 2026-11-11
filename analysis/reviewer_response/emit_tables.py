"""Emit camera-ready LaTeX for Table 2 (utilitarian) and appendix Tables 6-8.

One pinned log per model (FINAL), one scoring convention (fail-wrong: an
unparsed or refused answer counts as incorrect, denominator = all scenarios in
the game), targets from the current data/gt-harmbench-cleaned.csv (corrected
NSW). Formatting mirrors the existing Overleaf tables: RdYlGn cell colours,
bold column maxima in Table 2, unweighted game mean in the appendix Avg row.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.emit_tables > tables.tex
"""
from __future__ import annotations

import argparse

from matplotlib import colormaps, colors

from analysis.reviewer_response.reproduce_tables import CLEAN, GAMES, score

# the 15 models in the paper, in table order; logs pinned in reproduce_tables.LOGS
FINAL = {m: m for m in [
    "Claude 4.5 Opus", "Claude 4.5 Sonnet", "GPT-5.2", "GPT-5.1", "GPT-5 Mini", "GPT-5 Nano",
    "GPT-4o", "Grok 4.1 Fast", "Gemini 3 Pro", "Gemini 3 Flash", "Llama 3.3 70B",
    "Llama 3.2 3B", "Qwen3 30B", "Qwen3 8B", "DeepSeek V3.2"]}
GAME_TEX = {"Prisoner's Dilemma": "Prisoner's Dilemma", "Chicken": "Chicken",
            "Bach or Stravinski": "Battle of the Sexes", "Stag hunt": "Stag hunt",
            "Coordination": "Coordination", "No conflict": "No conflict"}
CMAP = colormaps["RdYlGn"]


def cell(v, lo=0.0, hi=1.0, bold=False):
    hexc = colors.to_hex(CMAP(lo + (hi - lo) * v))[1:].upper()
    num = f"{v:.2f}"
    return f"\\cellcolor[HTML]{{{hexc}}}" + (f"\\textbf{{{num}}}" if bold else num)


def table2(res):
    keys = GAMES + ["wavg"]
    rows = {m: [round(res[FINAL[m]]["util"][k], 2) for k in keys] for m in FINAL}
    best = [max(r[i] for r in rows.values()) for i in range(len(keys))]
    out = []
    for m, r in rows.items():
        out.append(f"{m} & " + " & ".join(cell(v, bold=v == best[i]) for i, v in enumerate(r)) + " \\\\")
    avg = [sum(res[FINAL[m]]["util"][k] for m in FINAL) / len(FINAL) for k in keys]
    out.append("\\midrule")
    out.append("\\textbf{Average} & " + " & ".join(cell(v) for v in avg) + " \\\\")
    return "\n".join(out)


def appendix(res, met):
    lo, hi = 0.1, 0.9
    out = []
    for g in GAMES + ["avg"]:
        vals = [res[FINAL[m]][met][g] for m in FINAL]
        vals.append(sum(vals) / len(vals))
        if g == "avg":
            out.append("\\midrule")
            out.append("\\textbf{Avg} & " + " & ".join(cell(v, lo, hi, bold=True) for v in vals) + " \\\\")
        else:
            out.append(f"{GAME_TEX[g]} & " + " & ".join(cell(v, lo, hi) for v in vals) + " \\\\")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(CLEAN))
    a = ap.parse_args()
    res = score(a.targets, fail_wrong=True)
    print("% Table 2 (utilitarian) body rows -- 3-result.tex")
    print(table2(res))
    for met, label in [("rawls", "Table 6 Rawlsian"), ("nash", "Table 7 Nash equilibrium"),
                       ("nsw", "Table 8 Nash social welfare (corrected own-min NSW)")]:
        print(f"\n% {label} body rows -- app_additional_results.tex")
        print(appendix(res, met))


if __name__ == "__main__":
    main()
