"""Extract per-model chosen action profiles from the headline baseline logs.

Read-only. Reads logs/baseline/*.eval (each run on the unfiltered 2265-row
gt-harmbench-targets set) and maps every model's chosen action profile onto the
canonical clean 1535-instance set.

Join strategy (robust to log-version drift):
  1. exact story_row match  -> clean id            (works for 14/15 models)
  2. else (actions_row, actions_column, payoff-matrix) singleton key -> clean id
     (gpt-5-mini's logs use a different story-text revision but identical games)

Choice parsing is tolerant of markdown (`**ANSWER: go bold**`) and of answers
where an explanation was concatenated after the label: a choice string is mapped
to an action index by longest-prefix match against that player's action labels.

Output: analysis/reviewer_response/data/model_choices.csv
Columns: model, instance_id, formal_game, row_idx, col_idx, status, join
status in {ok, refusal, unparsed, unmapped, nojoin}
"""
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

from inspect_ai.log import read_eval_log

REPO = Path(__file__).resolve().parents[2]
CLEAN_CSV = REPO / "data" / "gt-harmbench-cleaned.csv"
LOG_DIR = REPO / "logs" / "baseline"
OUT = Path(__file__).resolve().parent / "data" / "model_choices.csv"

ANS_RE = re.compile(r"Row choice:\s*(.*?),\s*Column choice:\s*(.*)", re.I | re.S)
CELLS = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]


def norm(x: str) -> str:
    """lowercase, strip markdown asterisks and surrounding punctuation/space."""
    return re.sub(r"[*`]+", "", str(x)).strip().lower().strip(" .:\n\"'")


def as_actions(v):
    if isinstance(v, list):
        return [norm(a) for a in v]
    v = v.strip()
    if v.startswith("["):
        return [norm(a) for a in ast.literal_eval(v)]
    return [norm(a) for a in v.split(",")]


def map_choice(raw: str, actions: list[str]) -> int | None:
    """Map a (possibly noisy) choice string to an action index.

    Exact match first, then longest action label that the choice starts with,
    then longest action label contained in the choice's first line, then an
    abbreviation that is the trailing word(s) of exactly one action.
    """
    r = norm(raw)
    if r in actions:
        return actions.index(r)
    first_line = r.split("\n", 1)[0]
    cands = [(len(a), i) for i, a in enumerate(actions) if r.startswith(a)]
    if cands:
        return max(cands)[1]
    cands = [(len(a), i) for i, a in enumerate(actions) if a and a in first_line]
    if cands:
        return max(cands)[1]
    # abbreviated answer ("beta" for "choose beta", "tighten" for "tighten
    # safeguards"): the leading or trailing word(s) of exactly one action. The
    # eval scorer accepts these via its substring check; without this, Llama 3.2
    # 3B loses ~90 valid answers and its Table 2 row does not reproduce.
    words = first_line.strip(" .*").split()
    if words:
        n = len(words)
        hits = [i for i, a in enumerate(actions)
                if a.split()[-n:] == words or a.split()[:n] == words]
        if len(hits) == 1:
            return hits[0]
    return None


def payoff_key(actions_row, actions_col, cells):
    ar = tuple(norm(a) for a in actions_row)
    ac = tuple(norm(a) for a in actions_col)
    return (ar, ac, tuple(tuple(c) for c in cells))


def build_clean_lookup(rows):
    by_story = {r["story_row"]: r["id"] for r in rows}
    key_ids: dict = {}
    for r in rows:
        cells = [ast.literal_eval(r[k]) for k in CELLS]
        ar = as_actions(r["actions_row"])
        ac = as_actions(r["actions_column"])
        key_ids.setdefault(payoff_key(ar, ac, cells), []).append(r["id"])
    singleton = {k: v[0] for k, v in key_ids.items() if len(v) == 1}
    return by_story, singleton


def sample_cells(md):
    rm = md.get("rewards_matrix")
    rm = rm if isinstance(rm, list) else ast.literal_eval(rm)
    # rm = [[c11, c12], [c21, c22]]; flatten to [c11, c12, c21, c22]
    return [rm[0][0], rm[0][1], rm[1][0], rm[1][1]]


def main() -> None:
    rows = list(csv.DictReader(open(CLEAN_CSV)))
    game_by_id = {r["id"]: r["formal_game"] for r in rows}
    by_story, singleton = build_clean_lookup(rows)

    out_rows = []
    summary = []
    for log_path in sorted(LOG_DIR.glob("*.eval")):
        model = log_path.stem
        log = read_eval_log(str(log_path))
        counts = {"ok": 0, "refusal": 0, "unparsed": 0, "unmapped": 0, "nojoin": 0}
        for s in log.samples:
            md = s.metadata or {}
            # ---- join to a clean instance id ----
            cid = by_story.get(md.get("story_row"))
            join = "story"
            if cid is None:
                cells = sample_cells(md)
                cid = singleton.get(
                    payoff_key(md["actions_row"], md["actions_column"], cells)
                )
                join = "payoff"
            if cid is None:
                counts["nojoin"] += 1
                continue
            # ---- parse the chosen profile ----
            sc = list(s.scores.values())[0]
            ans = getattr(sc, "answer", None) or ""
            m = ANS_RE.search(ans)
            status, ri, ci = "ok", "", ""
            if not m:
                status = "unparsed"
            else:
                ar = as_actions(md["actions_row"])
                ac = as_actions(md["actions_column"])
                ri_ = map_choice(m.group(1), ar)
                ci_ = map_choice(m.group(2), ac)
                if ri_ is not None and ci_ is not None:
                    ri, ci = ri_, ci_
                else:
                    # long free-text with no recoverable label = safety refusal
                    long = len(norm(m.group(1))) > 60 or len(norm(m.group(2))) > 60
                    status = "refusal" if long else "unmapped"
            counts[status] += 1
            out_rows.append(
                dict(model=model, instance_id=cid, formal_game=game_by_id[cid],
                     row_idx=ri, col_idx=ci, status=status, join=join)
            )
        summary.append((model, counts))
        print(f"{model:20s} " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
