"""Per-(model, game) run-to-run SD from the k=5 repeated-sampling study, for the
± values in every cell of Table 2. Same scoring as variance_analysis.py.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.variance_per_game
"""
from __future__ import annotations

import collections, glob, json, statistics as st
from pathlib import Path

from inspect_ai.log import read_eval_log

from analysis.reviewer_response.variance_analysis import score_sample, DISPLAY

HERE = Path(__file__).resolve().parent
GAMES = ["Prisoner's Dilemma", "Chicken", "Bach or Stravinski", "Stag hunt", "Coordination", "No conflict"]


def main():
    out = {}
    for path in sorted(glob.glob("logs/variance/**/*.eval", recursive=True)):
        log = read_eval_log(path)
        if log.status != "success" or not log.samples:
            continue
        name = DISPLAY.get(log.eval.model.split("/")[-1], log.eval.model.split("/")[-1])
        per = collections.defaultdict(lambda: collections.defaultdict(list))  # game -> epoch -> [ok]
        for s in log.samples:
            if "error during generation" in str(s.output.choices[0].message.content):
                continue
            ok = score_sample(s, "utilitarian")
            if ok is None:
                continue
            g = (s.metadata or {}).get("formal_game")
            per[g][s.epoch].append(ok); per["ALL"][s.epoch].append(ok)
        out[name] = {g: st.stdev([sum(v) / len(v) for v in per[g].values()]) for g in GAMES + ["ALL"] if per[g]}
    (HERE / "data" / "variance_per_game.json").write_text(json.dumps(out, indent=1))
    for m, d in out.items():
        print(f"{m:20s} " + " ".join(f"{g[:4]} {d.get(g, float('nan')):.3f}" for g in GAMES + ["ALL"]))


if __name__ == "__main__":
    main()
