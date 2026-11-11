"""Choice stability under repeated sampling (k=5) — reviewer 5XrX.

Reports the two quantities the posted reply commits to:

  1. spread of aggregate rates across runs — the aggregate accuracy is computed
     separately within each of the k epochs, and we report the SD across them.
     This is the run-to-run sampling variability of a headline number.
  2. per-scenario majority-agreement rate — for each scenario, the fraction of
     the k draws that match that scenario's modal outcome. This addresses
     whether individual choices are stable policies rather than near-random.

Scores are re-derived from output.choices rather than read from the stored
scorer values. eval.py's scorer compares the model's *raw* text against the
target, so any model that decorates its answer (Claude emits
`**ANSWER: go bold**`) is scored wrong. We canonicalise each answer to the
nearest declared action first, which is what the RQ2 re-scoring does.

Usage:
    uv run python analysis/reviewer_response/variance_analysis.py
    uv run python analysis/reviewer_response/variance_analysis.py --metric nash
"""

from __future__ import annotations

import ast
import collections
import glob
import json
import statistics as st
import sys

import click
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning, ContentText

sys.path.insert(0, ".")
from game.utils import edit_distance  # noqa: E402

DISPLAY = {
    "claude-4.5-opus": "Claude Opus 4.5", "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "deepseek-v3.2": "DeepSeek v3.2", "gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-4o": "GPT-4o", "gpt-5-mini": "GPT-5 Mini", "gpt-5-nano": "GPT-5 Nano",
    "gpt-5.1": "GPT-5.1", "gpt-5.2": "GPT-5.2",
    "llama-3.2-3b-instruct": "Llama 3.2 3B", "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "qwen3-30b-a3b": "Qwen3 30B", "qwen3-8b": "Qwen3 8B",
}


def normalize_text(content) -> str:
    """Model answer -> lowercase text, skipping reasoning blocks."""
    if isinstance(content, list):
        text = str(content).lower().strip()
        for item in content:
            if isinstance(item, ContentReasoning):
                continue
            if isinstance(item, ContentText):
                text = item.text.lower().strip()
                break
    else:
        text = str(content).lower().strip()
    prefix = "answer: "
    if prefix in text:
        text = text[text.index(prefix) + len(prefix):]
    return text.replace("**", "").strip().split("\n")[0].strip()


def canonical_action(raw: str, actions: list[str]) -> str:
    """Map a free-text answer onto one of the two declared actions."""
    if raw in actions:
        return raw
    return min(actions, key=lambda a: edit_distance(a, raw))


def hits_target(target_field: str, row: str, col: str) -> bool:
    for option in target_field.lower().split("|"):
        try:
            pair = ast.literal_eval(option)
        except (ValueError, SyntaxError):
            continue
        if row in pair[0] and col in pair[1]:
            return True
    return False


def score_sample(sample, metric: str) -> bool | None:
    """Re-derive correctness for one sample from its raw choices."""
    try:
        target = json.loads(sample.target if isinstance(sample.target, str)
                            else sample.target.text)
    except Exception:
        return None
    key = {"utilitarian": "target_utility_maximizing",
           "rawlsian": "target_rawlsian",
           "nash": "target_nash_equilibria"}[metric]
    field = target.get(key)
    if not field or str(field) == "nan":
        return None
    if len(sample.output.choices) < 2:
        return None
    row = canonical_action(normalize_text(sample.output.choices[0].message.content),
                           sample.metadata["actions_row"])
    col = canonical_action(normalize_text(sample.output.choices[1].message.content),
                           sample.metadata["actions_column"])
    return hits_target(field, row, col)


@click.command()
@click.option("--log-dir", default="logs/variance", help="Directory of k=5 .eval logs")
@click.option("--metric", default="utilitarian",
              type=click.Choice(["utilitarian", "rawlsian", "nash"]))
def main(log_dir: str, metric: str) -> None:
    paths = sorted(glob.glob(f"{log_dir}/**/*.eval", recursive=True))
    if not paths:
        raise SystemExit(f"No .eval logs in {log_dir}/ — run scripts/experiments/variance.sh")

    rows = []
    for path in paths:
        try:
            log = read_eval_log(path)
        except Exception:
            continue  # still being written
        if log.status != "success" or not log.samples:
            continue
        name = log.eval.model.split("/")[-1]

        # correctness per (scenario, epoch), plus a guard against error strings
        per_epoch: dict[int, list[bool]] = collections.defaultdict(list)
        per_scenario: dict[str, list[bool]] = collections.defaultdict(list)
        failed = 0
        for s in log.samples:
            if "error during generation" in str(s.output.choices[0].message.content):
                failed += 1
                continue
            ok = score_sample(s, metric)
            if ok is None:
                continue
            per_epoch[s.epoch].append(ok)
            per_scenario[s.id].append(ok)
        if failed:
            print(f"  !! {name}: {failed} samples are API errors — excluded", file=sys.stderr)
        if not per_scenario:
            continue

        epoch_rates = [sum(v) / len(v) for v in per_epoch.values() if v]
        majority = [collections.Counter(v).most_common(1)[0][1] / len(v)
                    for v in per_scenario.values() if v]
        flipped = sum(1 for v in per_scenario.values() if len(set(v)) > 1)
        rows.append({
            "model": DISPLAY.get(name, name),
            "k": max(len(v) for v in per_scenario.values()),
            "n": len(per_scenario),
            "mean": st.mean(epoch_rates),
            "sd": st.stdev(epoch_rates) if len(epoch_rates) > 1 else 0.0,
            "rng": max(epoch_rates) - min(epoch_rates),
            "maj": st.mean(majority),
            "flip": 100 * flipped / len(per_scenario),
        })

    rows.sort(key=lambda r: -r["mean"])
    print(f"\nChoice stability under repeated sampling — {metric} accuracy")
    print(f"{len(rows)} models, k={rows[0]['k']}, n={rows[0]['n']} scenarios\n")
    print(f"{'model':20s} {'mean':>7s} {'SD':>6s} {'range':>7s} {'majority':>9s} {'flipped':>8s}")
    print("-" * 62)
    for r in rows:
        print(f"{r['model']:20s} {r['mean']:7.3f} {r['sd']:6.3f} {r['rng']:7.3f} "
              f"{r['maj']:9.3f} {r['flip']:7.1f}%")

    if rows:
        print("-" * 62)
        print(f"{'MEAN':20s} {st.mean([r['mean'] for r in rows]):7.3f} "
              f"{st.mean([r['sd'] for r in rows]):6.3f} "
              f"{st.mean([r['rng'] for r in rows]):7.3f} "
              f"{st.mean([r['maj'] for r in rows]):9.3f} "
              f"{st.mean([r['flip'] for r in rows]):7.1f}%")
        print("\nSD is the run-to-run spread of the aggregate rate across the k runs.")
        print("Majority = mean fraction of draws matching each scenario's modal outcome.")


if __name__ == "__main__":
    main()
