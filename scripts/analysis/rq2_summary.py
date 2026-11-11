"""
RQ2 summary with correct filtering AND universal re-scoring from output.choices.
Re-scoring strips '**' and trailing reasoning, matching the original scorer logic.
Applied to ALL models for consistency (GPT-4o verified: 0 mismatches with stored scores).
Models with old list-format targets (GPT-5.1, GPT-5 Nano) fall back to stored scores.
"""
import ast
import glob
import json
import os
import sys

sys.path.insert(0, ".")
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

# Sample ids in the logs are in the original 2,265-row id space, which the
# released 1,535-row dataset does not preserve (its ids are renumbered 0..1534).
# So the clean subset cannot be derived from data/ and has to be supplied as a
# JSON list of log-space ids. Loaded lazily so importing this module never fails.
CLEAN_IDS_PATH = os.environ.get("CLEAN_IDS_PATH", "artifacts/data/clean_full_ids.json")
_CLEAN_IDS: set[int] | None = None


def clean_ids() -> set[int]:
    global _CLEAN_IDS
    if _CLEAN_IDS is None:
        if not os.path.exists(CLEAN_IDS_PATH):
            raise SystemExit(
                f"Missing {CLEAN_IDS_PATH}: a JSON list of the clean-subset sample ids\n"
                "in log id space. These cannot be recovered from the released dataset\n"
                "(see comment above). Set CLEAN_IDS_PATH to point at your own copy, or\n"
                "run with filter_clean=False to score every sample in the logs."
            )
        with open(CLEAN_IDS_PATH) as f:
            _CLEAN_IDS = set(json.load(f))
    return _CLEAN_IDS


def normalize_text(content) -> str:
    if isinstance(content, list):
        for item in content:
            if isinstance(item, ContentReasoning): continue
            if isinstance(item, ContentText):
                text = item.text.lower().strip(); break
        else:
            text = str(content).lower().strip()
    else:
        text = str(content).lower().strip()
    prefix = "answer: "
    if prefix in text:
        text = text[text.index(prefix) + len(prefix):]
    return text


def clean_action(text: str) -> str:
    text = text.replace("**", "").strip()
    if "\n" in text:
        text = text.split("\n")[0].strip()
    return text


def check_correctness(target_list, row_text, column_text) -> bool:
    for it in target_list:
        try: tv = ast.literal_eval(it)
        except: return False
        if row_text in tv[0] and column_text in tv[1]: return True
    return False


def score_from_output(s) -> dict | None:
    """Re-derive accuracy metrics from s.output.choices with clean_action fix.

    Handles two target formats:
    - JSON string (new format): json.loads(s.target) → dict with keys
    - list (old format): [nash, util, rawls, max_util, max_rawls]
    """
    target = s.target
    if isinstance(target, list):
        # Old format: use stored scores (these models have negligible ** artifacts)
        v = s.scores.get("all_strategies_scorer", {})
        if not v: return None
        v = v.value if hasattr(v, "value") else v
        return {k: v.get(k, float("nan")) for k in ["utilitarian", "nash"]}

    try:
        t = json.loads(target if isinstance(target, str) else target.text)
    except Exception:
        return None

    times = len(s.output.choices) // 2
    if times == 0:
        return None

    nash_target  = t.get("target_nash_equilibria","none").lower().strip()
    nash_target  = nash_target.split("|") if nash_target and str(nash_target) != "nan" else ["none"]
    util_target  = t["target_utility_maximizing"].lower().split("|")
    rawls_target = t["target_rawlsian"].lower().split("|")

    count_nash = count_util = count_rawls = 0
    for i in range(times):
        rt = clean_action(normalize_text(s.output.choices[2*i].message.content))
        ct = clean_action(normalize_text(s.output.choices[2*i+1].message.content))
        # Use edit distance to map cleaned text to canonical action name
        try: ri = s.metadata["actions_row"].index(rt)
        except ValueError:
            ri = min(range(2), key=lambda idx: edit_distance(s.metadata["actions_row"][idx], rt))
        try: ci = s.metadata["actions_column"].index(ct)
        except ValueError:
            ci = min(range(2), key=lambda idx: edit_distance(s.metadata["actions_column"][idx], ct))
        row_text = s.metadata["actions_row"][ri]
        col_text = s.metadata["actions_column"][ci]

        if check_correctness(nash_target, row_text, col_text) or nash_target == ["none"]:
            count_nash += 1
        if check_correctness(util_target, row_text, col_text):
            count_util += 1
        if check_correctness(rawls_target, row_text, col_text):
            count_rawls += 1

    return {"utilitarian": count_util / times, "nash": count_nash / times}


def canon(n): return CANONICAL.get(n.strip().lower(), n.strip())
def short(m):
    s = m.split("/")[-1]
    return MODEL_ALIASES.get(s, s)


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


# ── Load and filter ──────────────────────────────────────────────────────────
abstract  = load_dir("logs/abstract",  filter_clean=False, rescore=False)
narrative = load_dir("logs/baseline",  filter_clean=True,  rescore=True)
abstract.pop("gemini-3-pro-preview", None)
matched = sorted(set(abstract) & set(narrative))

if not matched:
    raise SystemExit(
        "No models found in both logs/abstract/ and logs/baseline/.\n"
        "This script recomputes the RQ2 summary from evaluation logs, which are not\n"
        "distributed with the repository. To produce them:\n"
        "  ./scripts/experiments/standard.sh   -> logs/baseline/\n"
        "  ./scripts/experiments/abstract.sh   -> logs/abstract/"
    )

def wagg(pg, metric):
    return sum(GT_WEIGHTS[g] * pg.loc[g, metric] for g in GT_COUNTS if g in pg.index)

print(f"Matched models: {len(matched)}\n")
print(f"{'Model':<22} {'n_narr':>7}  {'Narr util':>9} {'Abs util':>9} {'Δutil':>7}  {'Narr nash':>9} {'Abs nash':>9} {'Δnash':>7}")
print("-"*97)

model_rows = []
for model in matched:
    narr_df = narrative[model]
    abs_df  = abstract[model]
    npg = narr_df.groupby("game")[["utilitarian","nash"]].mean()
    apg = abs_df.groupby("game")[["utilitarian","nash"]].mean()
    name = DISPLAY.get(model, model)
    n_narr = len(narr_df)
    nw = {m: wagg(npg, m) for m in ["utilitarian","nash"]}
    aw = {m: wagg(apg, m) for m in ["utilitarian","nash"]}
    print(f"{name:<22} {n_narr:>7}  "
          f"{nw['utilitarian']:>9.3f} {aw['utilitarian']:>9.3f} {aw['utilitarian']-nw['utilitarian']:>+7.3f}  "
          f"{nw['nash']:>9.3f} {aw['nash']:>9.3f} {aw['nash']-nw['nash']:>+7.3f}")
    model_rows.append({"model": name,
                       "narr_util": nw["utilitarian"], "abs_util": aw["utilitarian"],
                       "narr_nash": nw["nash"],         "abs_nash": aw["nash"]})

mdf = pd.DataFrame(model_rows)
print("-"*97)
print(f"{'MEAN':<22} {'':>7}  "
      f"{mdf['narr_util'].mean():>9.3f} {mdf['abs_util'].mean():>9.3f} "
      f"{(mdf['abs_util']-mdf['narr_util']).mean():>+7.3f}  "
      f"{mdf['narr_nash'].mean():>9.3f} {mdf['abs_nash'].mean():>9.3f} "
      f"{(mdf['abs_nash']-mdf['narr_nash']).mean():>+7.3f}")
print(f"{'STD':<22} {'':>7}  "
      f"{mdf['narr_util'].std():>9.3f} {mdf['abs_util'].std():>9.3f} "
      f"{(mdf['abs_util']-mdf['narr_util']).std():>7.3f}  "
      f"{mdf['narr_nash'].std():>9.3f} {mdf['abs_nash'].std():>9.3f} "
      f"{(mdf['abs_nash']-mdf['narr_nash']).std():>7.3f}")

print("\n=== Per-game breakdown (mean Δutil across models) ===")
print(f"{'Game':<25} {'w':>6}  {'Narr':>8} {'Abs':>8} {'Δ':>8}")
print("-"*57)
for g in GT_COUNTS:
    ns  = [narrative[m].groupby("game")["utilitarian"].mean().get(g, float("nan")) for m in matched]
    as_ = [abstract[m].groupby("game")["utilitarian"].mean().get(g, float("nan"))  for m in matched]
    n_mean = np.nanmean(ns); a_mean = np.nanmean(as_)
    print(f"{g:<25} {GT_WEIGHTS[g]:>6.3f}  {n_mean:>8.3f} {a_mean:>8.3f} {a_mean-n_mean:>+8.3f}")
