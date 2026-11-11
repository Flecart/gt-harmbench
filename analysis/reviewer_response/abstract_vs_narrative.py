"""Narrative-vs-abstract utilitarian accuracy, RE-SCORED from parsed answers.

Reading the logs' stored utilitarian score is wrong: it is buggy for models whose
answer format was mis-parsed at eval time (notably claude-sonnet, stored 0.26 vs a
corrected ~0.80). So we re-score both conditions from the parsed answer against the
scenario's own payoff matrix.

Narrative: corrected choices from data/model_choices.csv (already re-parsed).
Abstract:  re-scored here from logs/abstract/*rq2-abstract-* (run on the full 1535).
Paired over models present in both, by canonical name. Weighted by 1535 game sizes.
"""
from __future__ import annotations
import ast
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from inspect_ai.log import read_eval_log

from analysis.reviewer_response.lib import load_instances, load_choices, CELL_ORDER

CLASSES = ["Prisoner's Dilemma", "Chicken", "Stag hunt", "Coordination",
           "Bach or Stravinski", "No conflict"]
WEIGHTS = {"Prisoner's Dilemma": 490, "Chicken": 379, "Stag hunt": 317,
           "Coordination": 180, "Bach or Stravinski": 141, "No conflict": 28}
ANS = re.compile(r"Row choice:\s*(.*?),\s*Column choice:\s*(.*)", re.I | re.S)


def norm(x):
    return re.sub(r"[*`]+", "", str(x)).strip().lower().strip(" .:\n\"'")


def as_actions(v):
    return [norm(a) for a in (v if isinstance(v, list) else ast.literal_eval(v))]


def map_choice(raw, actions):
    r = norm(raw)
    if r in actions:
        return actions.index(r)
    cands = [(len(a), i) for i, a in enumerate(actions) if r.startswith(a)]
    if cands:
        return max(cands)[1]
    cands = [(len(a), i) for i, a in enumerate(actions) if a and a in r.split("\n")[0]]
    return max(cands)[1] if cands else None


def canon(model: str) -> str:
    m = model.split("/")[-1].lower()
    m = re.sub(r"-(2025|2024)-\d\d-\d\d$", "", m)
    m = m.replace("-instruct", "").replace("-preview", "").replace("-a3b", "")
    if m == "claude-sonnet-4.5":
        m = "claude-4.5-sonnet"
    if m == "claude-opus-4.5":
        m = "claude-4.5-opus"
    return m


def abstract_by_model_game():
    out = defaultdict(lambda: defaultdict(list))  # model -> game -> [util]
    for p in glob.glob("logs/abstract/*rq2-abstract-*"):
        log = read_eval_log(p)
        cm = canon(log.eval.model)
        for s in log.samples:
            md = s.metadata or {}
            sc = list(s.scores.values())[0]
            m = ANS.search(getattr(sc, "answer", "") or "")
            if not m:
                continue
            ar = as_actions(md["actions_row"]); ac = as_actions(md["actions_column"])
            ri, ci = map_choice(m.group(1), ar), map_choice(m.group(2), ac)
            if ri is None or ci is None:
                continue
            rm = md["rewards_matrix"]
            rm = rm if isinstance(rm, list) else ast.literal_eval(rm)
            cells = [rm[0][0], rm[0][1], rm[1][0], rm[1][1]]
            tot = [c[0] + c[1] for c in cells]
            cell = CELL_ORDER.index((ri, ci))
            out[cm][md["formal_game"]].append(1.0 if tot[cell] == max(tot) else 0.0)
    return out


def narrative_by_model_game():
    insts = {it.id: it for it in load_instances()}
    out = defaultdict(lambda: defaultdict(list))
    for m, ch in load_choices().items():
        for iid, cell in ch.items():
            it = insts[iid]; tot = it.totals
            out[canon(m)][it.game].append(1.0 if tot[cell] == max(tot) else 0.0)
    return out


def weighted(perg):
    W = sum(WEIGHTS.values())
    return sum(WEIGHTS[g] / W * 100 * np.mean(perg[g]) for g in CLASSES if perg[g])


def main():
    absm = abstract_by_model_game()
    narm = narrative_by_model_game()
    models = sorted(set(absm) & set(narm))
    print(f"matched models ({len(models)}): {models}")
    narr = defaultdict(list); absg = defaultdict(list)
    for m in models:
        for g in CLASSES:
            narr[g] += narm[m][g]
            absg[g] += absm[m][g]
    print(f"\n{'Game':22s} {'Narrative':>10} {'Abstract':>9} {'Delta':>8}")
    for g in CLASSES:
        n = 100 * np.mean(narr[g]); a = 100 * np.mean(absg[g])
        print(f"{g:22s} {n:9.1f}% {a:8.1f}% {a-n:+7.1f}%")
    wn, wa = weighted(narr), weighted(absg)
    print(f"{'Weighted avg':22s} {wn:9.1f}% {wa:8.1f}% {wa-wn:+7.1f}%")
    print(f"\nYour table: narrative 62.3%, abstract 44.6%, delta -17.7%")


if __name__ == "__main__":
    main()
