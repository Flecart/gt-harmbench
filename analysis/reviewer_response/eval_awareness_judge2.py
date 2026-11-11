"""Second-judge replication of the evaluation-awareness audit.

Re-labels the SAME traces that `eval_awareness_judge.py` already labelled with
Claude Haiku 4.5, using a different judge model (default Claude Sonnet 5). The
prompt, temperature, token cap and parsing are imported from that module rather
than restated, so the only thing that varies between the two runs is the judge.

Motivation: the audit's largest hole is that the judge was never validated —
precision and recall are unmeasured. A full human labelling is the ideal fix;
a second independent judge is the cheap approximation. Inter-judge agreement
puts a ceiling on how much of the 2.11% rate is judge idiosyncrasy.

Rows are selected by joining `judged.jsonl` back to `traces_k5.json` on
(model, scenario_id, epoch, role), so the two runs cover an identical set and
the comparison is paired.

Usage:
    uv run python analysis/reviewer_response/eval_awareness_judge2.py --dry-run
    uv run python analysis/reviewer_response/eval_awareness_judge2.py --limit 25
    uv run python analysis/reviewer_response/eval_awareness_judge2.py
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import click
from openai import OpenAI

from eval_awareness_judge import judge_one

ART = pathlib.Path("artifacts/eval_awareness")
TRACES = ART / "traces_k5.json"
BASELINE = ART / "judged.jsonl"
PROMPT_PATH = ART / "judge_prompt.txt"

# OpenRouter per-M-token prices, used only for the pre-flight cost estimate.
PRICING = {
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
}


def load_key() -> str:
    """Prefer the ambient env var; fall back to a .env file if present."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = pathlib.Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            s = line.strip().removeprefix("export ").strip()
            if s.startswith("#") or not s.startswith("OPENROUTER_API_KEY"):
                continue
            return s.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("OPENROUTER_API_KEY not found in environment or .env")


def rowkey(r: dict) -> tuple:
    return (r["model"], str(r["scenario_id"]), r["epoch"], r["role"])


def build_matched_set(limit: int | None, seed: int) -> list[dict]:
    """The trace records corresponding to every row in the baseline judgement.

    Carries `stratum` across from the baseline so the second run can be
    stratum-weighted the same way without re-deriving the prefilter.
    """
    traces = {rowkey(r): r for r in json.loads(TRACES.read_text())}
    out, missing = [], 0
    seen = set()
    for line in BASELINE.read_text().splitlines():
        if not line.strip():
            continue
        b = json.loads(line)
        k = rowkey(b)
        if k in seen:
            continue
        seen.add(k)
        rec = traces.get(k)
        if rec is None:
            missing += 1
            continue
        out.append({**rec, "stratum": b.get("stratum")})
    if missing:
        print(f"warning: {missing} baseline rows had no matching trace")
    if limit:
        # Stratified-ish smoke sample: shuffle so a --limit run spans models.
        random.Random(seed).shuffle(out)
        out = out[:limit]
    return out


@click.command()
@click.option("--judge-model", default="anthropic/claude-sonnet-5")
@click.option("--out", default=None, help="Defaults to judged_<model-slug>.jsonl")
@click.option("--limit", default=0, type=int, help="Smoke-test on N rows (0 = all)")
@click.option("--seed", default=42)
@click.option("--concurrency", default=12)
@click.option("--max-tokens", default=1100,
              help="Raise for judges whose reasoning tokens starve the response")
@click.option("--max-cost", default=30.0, help="Abort if the estimate exceeds this")
@click.option("--dry-run", is_flag=True, help="Print the plan and cost, call nothing")
def main(judge_model, out, limit, seed, concurrency, max_tokens, max_cost, dry_run):
    sample = build_matched_set(limit or None, seed)
    slug = judge_model.split("/")[-1].replace(".", "")
    out_path = pathlib.Path(out) if out else ART / f"judged_{slug}.jsonl"

    chars = sum(len(r["story"][:6000]) + len(r["reasoning"][:12000]) for r in sample)
    in_tok = chars / 3.8 + 1250 * len(sample)
    p_in, p_out = PRICING.get(judge_model, (2.00, 10.00))
    cost = in_tok / 1e6 * p_in + 250 * len(sample) / 1e6 * p_out
    n_hit = sum(1 for r in sample if r["stratum"] == "prefilter_hit")
    print(f"baseline          : {BASELINE}")
    print(f"to re-judge       : {len(sample)}  ({n_hit} prefilter hits + "
          f"{len(sample) - n_hit} random non-hits)")
    print(f"judge model       : {judge_model}")
    print(f"output            : {out_path}")
    print(f"est. input tokens : {in_tok / 1e6:.2f}M")
    print(f"EST COST          : ${cost:,.2f}")
    if dry_run:
        return
    if cost > max_cost:
        raise SystemExit(f"Cost estimate exceeds ${max_cost:.2f} — stopping. "
                         f"Raise --max-cost to override.")

    prompt = PROMPT_PATH.read_text()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())

    # Only successful rows count as done. The baseline harness resumed on any
    # written row, which permanently strands anything that failed transiently
    # (rate limit, auth, truncation) — a retry would skip it.
    done, failed = set(), 0
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("ok"):
                done.add(rowkey(d))
            else:
                failed += 1
        print(f"resuming: {len(done)} already judged"
              + (f" ({failed} earlier failures will be retried)" if failed else ""))
    todo = [r for r in sample if rowkey(r) not in done]
    if not todo:
        print("nothing to do")
        return

    lock = threading.Lock()
    counter = {"n": 0}
    with out_path.open("a") as fh, ThreadPoolExecutor(concurrency) as pool:
        for res in pool.map(
                lambda r: judge_one(client, judge_model, prompt, r, max_tokens), todo):
            with lock:
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                counter["n"] += 1
                if counter["n"] % 200 == 0:
                    print(f"  judged {counter['n']}/{len(todo)}", flush=True)
    print(f"done — wrote {out_path}")


if __name__ == "__main__":
    main()
