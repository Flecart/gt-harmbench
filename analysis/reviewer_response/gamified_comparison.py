"""RQ2 gamified comparison: prosaic (RQ1 baseline logs) vs numerical framing
(payoffs shown, logs/RQ2 - gamified/), on the 1,535 release.

Matches the paper figure (fig:gamify-variation): utilitarian and Nash accuracy,
weighted over Prisoner's Dilemma and Chicken, unparseable answers counted wrong.
The 24 scenarios whose payoffs were patched after the RQ2 runs (22 symmetry +
2 BoS) are Stag Hunt / BoS / No Conflict, so they never enter this figure.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.gamified_comparison
"""
from __future__ import annotations

import csv, json
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

from analysis.reviewer_response.extract_choices import (
    ANS_RE, as_actions, map_choice, build_clean_lookup, sample_cells, payoff_key)
from analysis.reviewer_response.reproduce_tables import CLEAN, CACHE, target_cells

HERE = Path(__file__).resolve().parent
G = Path("logs/RQ2 - gamified")
GCACHE = HERE / "data" / "gamified_choices.json"
GAMES = ["Prisoner's Dilemma", "Chicken"]
# paper label -> gamified log (the 10 Jan gamify-experiment set; 6 Jan Sonnet/Nano, 13-14 Jan Gemini Pro/Opus)
GAMIFIED = {
    "Claude 4.5 Opus": G / "2026-01-14T20-42-18+01-00_gamify-experiment-claude-4.5-opus_Tbdf4xTBWuvp9dkUmq286k.eval",
    "Claude 4.5 Sonnet": G / "2026-01-06T20-52-47+01-00_gamify-experiment-claude-sonnet-4.5_9RTR6DjjPByjBTX3uF4fjF.eval",
    "GPT-5.2": G / "2026-01-10T10-01-18+01-00_gamify-experiment-gpt-5.2_EUPrRMj2WK5UTJuK3PKMci.eval",
    "GPT-5.1": G / "2026-01-10T09-54-38+01-00_gamify-experiment-gpt-5.1_6PkNVovXrPH5oKfUxWUQ43.eval",
    "GPT-5 Mini": G / "2026-01-10T09-54-38+01-00_gamify-experiment-gpt-5-mini_axkuxz2aoTWCJvXSfyzQrD.eval",
    "GPT-5 Mini (5 Jan gamified, was RQ1 baseline)": G / "2026-01-05_gpt-5-mini_gamify_formerly-baseline.eval",
    "GPT-5 Nano": G / "2026-01-06T20-52-47+01-00_gamify-experiment-gpt-5-nano_GTHnua7ULzNmj4rVeNf9Pg.eval",
    "GPT-4o": G / "2026-01-10T12-01-38+01-00_gamify-experiment-gpt-4o_DAJGqsRmqJTpSBELu8wW2z.eval",
    "Grok 4.1 Fast": G / "2026-01-10T16-11-49+01-00_gamify-experiment-grok-4.1-fast_G4AxiGcbnT5yb7cRxWEux4.eval",
    "Grok 4.1 Fast (12-08 run)": G / "2026-01-10T12-08-39+01-00_gamify-experiment-grok-4.1-fast_AYsjcG3SYtEbRDUdqAgedP.eval",
    "Gemini 3 Pro": G / "2026-01-13T17-32-22+01-00_gamify-experiment-gemini-3-pro-preview_msGAQ2hun9NtXJACnFcsWx.eval",
    "Gemini 3 Flash": G / "2026-01-10T10-12-12+01-00_gamify-experiment-gemini-3-flash-preview_eTxMqGX5itS5h47wizp6mi.eval",
    "Llama 3.3 70B": G / "2026-01-10T10-04-23+01-00_gamify-experiment-llama-3.3-70b-instruct_iN7RH7rqy7oAqDfvkQX8Ed.eval",
    "Llama 3.2 3B": G / "2026-01-10T10-04-31+01-00_gamify-experiment-llama-3.2-3b-instruct_R7XAQgnY938RiBhX2i7rvP.eval",
    "Qwen3 30B": G / "2026-01-10T10-09-20+01-00_gamify-experiment-qwen3-30b-a3b_o5AKiVFFYFJfMHiKryMuH5.eval",
    "Qwen3 8B": G / "2026-01-10T10-11-04+01-00_gamify-experiment-Qwen3-8B_nzHCe3tzN2x5TKprnaJNjZ.eval",
}


def extract():
    rows = list(csv.DictReader(open(CLEAN)))
    by_story, singleton = build_clean_lookup(rows)
    cache = json.loads(GCACHE.read_text()) if GCACHE.exists() else {}
    for model, p in GAMIFIED.items():
        if model in cache:
            continue
        out = {}
        for s in read_eval_log(str(p)).samples:
            md = s.metadata or {}
            cid = singleton.get(payoff_key(md["actions_row"], md["actions_column"], sample_cells(md)))
            if cid is None or cid in out or not s.scores:
                continue
            m = ANS_RE.search(getattr(list(s.scores.values())[0], "answer", None) or "")
            if not m:
                continue
            ri = map_choice(m.group(1), as_actions(md["actions_row"]))
            ci = map_choice(m.group(2), as_actions(md["actions_column"]))
            if ri is not None and ci is not None:
                out[cid] = [ri, ci]
        cache[model] = out
        print(f"{model:46s} {len(out)} parsed", flush=True)
        GCACHE.write_text(json.dumps(cache))
    return cache


def pd_chicken(choices, rows):
    n = {g: sum(r["formal_game"] == g for r in rows.values()) for g in GAMES}
    hit = defaultdict(lambda: [0, 0])
    for cid, (ri, ci) in choices.items():
        r = rows[cid]
        if r["formal_game"] in GAMES:
            hit[r["formal_game"]][0] += (ri, ci) in target_cells(r, "target_utility_maximizing")
            hit[r["formal_game"]][1] += (ri, ci) in target_cells(r, "target_nash_equilibria")
    tot = sum(n.values())
    return sum(hit[g][0] for g in GAMES) / tot, sum(hit[g][1] for g in GAMES) / tot


def main():
    gam = extract()
    base = json.loads(CACHE.read_text())
    rows = {r["id"]: r for r in csv.DictReader(open(CLEAN))}
    print(f"\n{'model':46s} {'prosaic U':>9} {'N':>5}   {'numeric U':>9} {'N':>5}   {'dU':>6} {'dN':>6}")
    du, dn = [], []
    for m in GAMIFIED:
        bm = m.split(" (")[0]
        if bm not in base:
            continue
        u0, n0 = pd_chicken(base[bm], rows)
        u1, n1 = pd_chicken(gam[m], rows)
        print(f"{m:46s} {u0:9.2f} {n0:5.2f}   {u1:9.2f} {n1:5.2f}   {u1-u0:+6.2f} {n1-n0:+6.2f}")
        if "(" not in m:
            du.append(u1 - u0); dn.append(n1 - n0)
    print(f"\nmean over {len(du)} models: dU {sum(du)/len(du):+.4f}  dN {sum(dn)/len(dn):+.4f}")


if __name__ == "__main__":
    main()
