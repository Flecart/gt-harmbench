"""Reproduce the paper's per-model x per-game accuracy tables from the logs.

Table 2 (utilitarian, body), Tables 6-8 (Rawlsian, Nash equilibrium, NSW,
appendix). Choices are parsed from the baseline logs (same parser as
extract_choices.py), joined to the 1,535 release, and scored against the target
columns of a dataset CSV: the submitted one (old targets) or the current one
(corrected NSW). The paper's numbers are parsed from the Overleaf tex for
comparison.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.reproduce_tables --extract
  PYTHONPATH=. uv run python -m analysis.reviewer_response.reproduce_tables \
      --targets <submitted csv> --tex ~/Repos/gt-harmbench-overleaf/GTHarmBench
"""
from __future__ import annotations

import argparse, ast, csv, json, re
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

from analysis.reviewer_response.extract_choices import (
    ANS_RE, CELLS, as_actions, map_choice, build_clean_lookup, sample_cells, payoff_key)

REPO = Path(__file__).resolve().parents[2]
CLEAN = REPO / "data" / "gt-harmbench-cleaned.csv"
CACHE = Path(__file__).resolve().parent / "data" / "baseline_choices.json"
L = REPO / "logs"
# model label as printed in the paper -> log
LOGS = {
    "Claude 4.5 Opus": L / "baseline/claude-4.5-opus.eval",
    "Claude 4.5 Sonnet": L / "baseline/claude-4.5-sonnet.eval",
    "GPT-5.2": L / "baseline/gpt-5.2.eval",
    "GPT-5.1": L / "baseline/gpt-5.1.eval",
    "GPT-5 Mini": L / "baseline/gpt-5-mini.eval",
    "GPT-5 Nano": L / "baseline/gpt-5-nano.eval",
    "GPT-4o": L / "baseline/gpt-4o.eval",
    "Grok 4.1 Fast": L / "baseline/grok-4.1-fast.eval",
    "Gemini 3 Pro": L / "baseline/gemini-3-pro.eval",
    "Gemini 3 Flash": L / "baseline/gemini-3-flash.eval",
    "Llama 3.3 70B": L / "baseline/llama-3.3-70b.eval",
    "Llama 3.2 3B": L / "baseline/llama-3.2-3b.eval",
    "Qwen3 30B": L / "baseline/qwen3-30b.eval",
    "Qwen3 8B": L / "baseline/qwen3-8b.eval",
    "DeepSeek V3.2": L / "baseline/deepseek-v3.2.eval",
    "GPT-4o mini (not in paper)": L / "baseline/gpt-4o-mini.eval",
    # logs the submitted paper used before the camera-ready swap (see logs/baseline/README.md)
    "GPT-5 Mini (submitted: gamified run)": L / "RQ2 - gamified/2026-01-05_gpt-5-mini_gamify_formerly-baseline.eval",
    "GPT-5.1 (2 Jan run)": L / "baseline-superseded/gpt-5.1_2026-01-02.eval",
    "GPT-5 Nano (5 Jan run)": L / "baseline-superseded/gpt-5-nano_2026-01-05.eval",
}
GAMES = ["Prisoner's Dilemma", "Chicken", "Bach or Stravinski", "Stag hunt", "Coordination", "No conflict"]
METRICS = {"util": "target_utility_maximizing", "rawls": "target_rawlsian",
           "nash": "target_nash_equilibria", "nsw": "target_nash_social_welfare"}


def extract():
    rows = list(csv.DictReader(open(CLEAN)))
    by_story, singleton = build_clean_lookup(rows)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for model, p in LOGS.items():
        if model in cache:
            continue
        log = read_eval_log(str(p))
        out = {}
        for s in log.samples:
            md = s.metadata or {}
            cid = by_story.get(md.get("story_row"))
            if cid is None:
                cid = singleton.get(payoff_key(md["actions_row"], md["actions_column"], sample_cells(md)))
            if cid is None or cid in out:
                continue
            if not s.scores:
                continue
            m = ANS_RE.search(getattr(list(s.scores.values())[0], "answer", None) or "")
            if not m:
                continue
            ri = map_choice(m.group(1), as_actions(md["actions_row"]))
            ci = map_choice(m.group(2), as_actions(md["actions_column"]))
            if ri is not None and ci is not None:
                out[cid] = [ri, ci]
        cache[model] = out
        print(f"{model:28s} {len(out)} scored of 1535  ({log.eval.dataset.name})", flush=True)
        CACHE.write_text(json.dumps(cache))


def target_cells(r, col):
    ar = [a.lower() for a in ast.literal_eval(r["actions_row"])]
    ac = [a.lower() for a in ast.literal_eval(r["actions_column"])]
    s = {x.strip() for x in str(r[col]).lower().split("|")}
    return {(i, j) for i in range(2) for j in range(2) if str((ar[i], ac[j])) in s}


def score(targets_csv, fail_wrong=False):
    rows = {r["id"]: r for r in csv.DictReader(open(targets_csv))}
    game_n = defaultdict(int)
    for r in rows.values():
        game_n[r["formal_game"]] += 1
    cache = json.loads(CACHE.read_text())
    res = {}
    for model, ch in cache.items():
        res[model] = {}
        for met, col in METRICS.items():
            per = defaultdict(lambda: [0, 0])
            for cid, (ri, ci) in ch.items():
                r = rows[cid]
                per[r["formal_game"]][0] += 1
                per[r["formal_game"]][1] += (ri, ci) in target_cells(r, col)
            n = {k: (game_n[k] if fail_wrong else per[k][0]) for k in GAMES}
            g = {k: per[k][1] / n[k] for k in GAMES}
            g["avg"] = sum(g[k] for k in GAMES) / 6
            g["wavg"] = sum(g[k] * n[k] for k in GAMES) / sum(n.values())
            res[model][met] = g
    return res


NUM = re.compile(r"(\d\.\d\d)")


def parse_tex(texdir: Path):
    """Return {metric: {model: {game/avg: value}}} from the paper source."""
    out = {}
    body = (texdir / "3-result.tex").read_text().splitlines()
    # active Table 2: model rows (uncommented) with 7 numbers
    t2 = {}
    for ln in body:
        if ln.lstrip().startswith("%") or "&" not in ln:
            continue
        name = ln.split("&")[0].strip().replace("\\textbf{", "").rstrip("}")
        nums = NUM.findall(ln)
        if len(nums) == 7 and name in LOGS:
            t2[name] = dict(zip(GAMES + ["wavg"], map(float, nums)))
    out["util"] = t2
    app = (texdir / "app_additional_results.tex").read_text()
    for met, cap in [("rawls", "Rawlsian Accuracy"), ("nash", "Nash Accuracy across"), ("nsw", "Nash Social Accuracy")]:
        seg = app[app.index(cap):]
        seg = seg[: seg.index("end{tabular}")]
        lines = [l for l in seg.splitlines() if "&" in l and not l.lstrip().startswith("%")]
        header = [h.strip() for h in lines[0].split("&")[1:]]
        header = [re.sub(r"\\[a-z]+\{|\}|\\\\|\\scriptsize", "", h).strip() for h in header]
        tab = defaultdict(dict)
        for l in lines[1:]:
            cells = l.split("&")
            game = re.sub(r"\\[a-z]+\{|\}", "", cells[0]).strip()
            vals = [NUM.findall(c) for c in cells[1:]]
            for h, v in zip(header, vals):
                if v:
                    tab[h][game] = float(v[0])
        out[met] = dict(tab)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--targets", default=str(CLEAN))
    ap.add_argument("--tex")
    ap.add_argument("--fail-wrong", action="store_true",
                    help="count unparsed/refused answers as incorrect (denominator = all scenarios)")
    a = ap.parse_args()
    if a.extract:
        return extract()
    res = score(a.targets, a.fail_wrong)
    paper = parse_tex(Path(a.tex).expanduser()) if a.tex else {}
    for met in METRICS:
        print(f"\n## {met}  (ours, paper in brackets where available; avg = unweighted 6-game mean, wavg = weighted)")
        for model in LOGS:
            if model not in res:
                continue
            g = res[model][met]
            pm = paper.get(met, {})
            # appendix headers may abbreviate names
            pk = next((k for k in pm if k.replace(" ", "").lower() == model.replace(" ", "").lower()), None)
            pv = pm.get(pk, {}) if pk else {}
            def cell(k, pk_=None):
                v = f"{g[k]:.2f}"
                p = None
                for cand in ([pk_] if pk_ else []) + [k]:
                    if cand in pv:
                        p = pv[cand]
                return v + (f"[{p:.2f}]" if p is not None else "") + ("*" if p is not None and abs(p - g[k]) > 0.015 else "")
            short = {"Prisoner's Dilemma": "PD", "Chicken": "CH", "Bach or Stravinski": "BoS",
                     "Stag hunt": "SH", "Coordination": "CO", "No conflict": "NC"}
            print(f"{model:28s} " + " ".join(f"{short[k]} {cell(k)}" for k in GAMES)
                  + f"  avg {g['avg']:.2f}  wavg {cell('wavg')}")


if __name__ == "__main__":
    main()
