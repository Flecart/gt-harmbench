"""Rewrite the two NSW columns of a dataset CSV with the corrected Nash social
welfare (own-min disagreement point; game.metrics.nash_social_payoff).

Same computation as scripts/generation/add_targets.py, but touches only
`target_nash_social_welfare` and `nash_social_welfare` and writes with the csv
module, so every other byte of the file is unchanged (add_targets.py round-trips
through pandas and reformats). Verified that add_targets reproduces all other
target columns of gt-harmbench-cleaned.csv exactly.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.regenerate_nsw_targets data/gt-harmbench-cleaned.csv
"""
from __future__ import annotations

import ast, csv, sys
from collections import Counter

from game.metrics import nash_social_welfare
from scripts.generation.add_targets import get_targets

K = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]


def main(path: str) -> None:
    with open(path, newline="") as f:
        first = f.readline()
        f.seek(0)
        rows = list(csv.DictReader(f))
    eol = "\r\n" if first.endswith("\r\n") else "\n"  # keep the file's own line endings
    fields = list(rows[0].keys())
    changed = Counter()
    for r in rows:
        c = [ast.literal_eval(r[k]) for k in K]
        t = get_targets(r).nash_social_welfare
        new_t = "|".join(t) if t else "None"
        if set(new_t.split("|")) != set(r["target_nash_social_welfare"].split("|")):
            changed[r["formal_game"]] += 1
        r["target_nash_social_welfare"] = new_t
        r["nash_social_welfare"] = str(float(nash_social_welfare([[c[0], c[1]], [c[2], c[3]]])))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator=eol)
        w.writeheader(); w.writerows(rows)
    print(f"{path}: NSW optimal set changed in {sum(changed.values())}/{len(rows)}: {dict(changed)}")


if __name__ == "__main__":
    main(sys.argv[1])
