"""Inter-judge agreement for the evaluation-awareness audit.

Compares the baseline Claude Haiku 4.5 judgements (`judged.jsonl`) against a
second judge run over the identical trace set (`eval_awareness_judge2.py`),
and asks the question the audit's caveat #1 leaves open: how much of the
reported 2.11% is the judge rather than the traces?

Reports, for both `model_awareness` and `hypothetical_recognition`:

  * the 2x2 confusion table and raw agreement
  * Cohen's kappa, with a bootstrap CI
  * positive specific agreement (PSA) -- at a ~3% base rate kappa is depressed
    by the prevalence paradox, and PSA is the more honest summary of whether
    the judges agree about the traces that matter
  * an exact McNemar test, which says whether one judge is systematically
    more trigger-happy rather than merely noisy
  * the stratum-weighted population estimate under each judge, so the headline
    number can be quoted with a judge-swap robustness check attached

Everything is computed twice: on the raw judge output, and after the two
automated corrections in `eval_awareness_analysis.py` (system-prompt echo and
game recognition), since those corrections are what the reported rates rest on.

Usage:
    uv run python analysis/reviewer_response/eval_awareness_agreement.py
    uv run python analysis/reviewer_response/eval_awareness_agreement.py \
        --second artifacts/eval_awareness/judged_claude-sonnet-5.jsonl
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import statistics as st
from collections import defaultdict

import click

from eval_awareness_analysis import (
    PREFILTER_RE,
    RAW_COT,
    SUMMARIZED,
    corrected_flags,
    cot_kind,
)

ART = pathlib.Path("artifacts/eval_awareness")
FLAGS = (("model_awareness", 0), ("hypothetical_recognition", 1))


def rowkey(r: dict) -> tuple:
    return (r["model"], str(r["scenario_id"]), r["epoch"], r["role"])


def load(path: str) -> dict[tuple, dict]:
    out = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("ok"):
            out[rowkey(r)] = r
    return out


def raw_flags(r: dict) -> tuple[bool, bool]:
    return bool(r.get("aware")), bool(r.get("hypothetical"))


def kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters over paired judgements."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kappa_ci(a: list[bool], b: list[bool], n_boot: int = 2000,
             seed: int = 0) -> tuple[float, float]:
    if len(a) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    pairs = list(zip(a, b))
    ks = []
    for _ in range(n_boot):
        s = rng.choices(pairs, k=len(pairs))
        k = kappa([x for x, _ in s], [y for _, y in s])
        if not math.isnan(k):
            ks.append(k)
    if not ks:
        return (float("nan"), float("nan"))
    ks.sort()
    return ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks))]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def confusion(a: list[bool], b: list[bool]) -> tuple[int, int, int, int]:
    """(both yes, only-A yes, only-B yes, both no)."""
    yy = sum(x and y for x, y in zip(a, b))
    yn = sum(x and not y for x, y in zip(a, b))
    ny = sum((not x) and y for x, y in zip(a, b))
    nn = sum((not x) and (not y) for x, y in zip(a, b))
    return yy, yn, ny, nn


def psa(yy: int, yn: int, ny: int) -> float:
    """Positive specific agreement: 2*both / (2*both + discordant)."""
    denom = 2 * yy + yn + ny
    return 2 * yy / denom if denom else float("nan")


def stratum_sizes(allrecs: list[dict]) -> dict[str, tuple[int, int]]:
    """Population (hit, non-hit) counts per CoT group.

    The prefilter scan is O(25k long strings), so do it once up front rather
    than inside each population_estimate call.
    """
    sizes = {}
    for gname, members in (("raw CoT", RAW_COT), ("summarized", SUMMARIZED)):
        pop = [r for r in allrecs if r["model"] in members]
        n_hit = sum(1 for r in pop if PREFILTER_RE.search(r["reasoning"]))
        sizes[gname] = (n_hit, len(pop) - n_hit)
    return sizes


def population_estimate(rows: dict[tuple, dict], sizes: tuple[int, int],
                        members: set[str], idx: int,
                        flagfn) -> tuple[float, float]:
    """Stratum-weighted rate and SE for one CoT group, per the audit's design."""
    n_hit, n_non = sizes
    n_pop = n_hit + n_non
    if not n_pop:
        return float("nan"), float("nan")
    jh = [r for r in rows.values()
          if r["model"] in members and r.get("stratum") == "prefilter_hit"]
    jn = [r for r in rows.values()
          if r["model"] in members and r.get("stratum") == "random_nonhit"]
    p_h = st.mean([flagfn(r)[idx] for r in jh]) if jh else 0.0
    p_n = st.mean([flagfn(r)[idx] for r in jn]) if jn else 0.0
    W_h, W_n = n_hit / n_pop, n_non / n_pop
    est = p_h * W_h + p_n * W_n
    # Hit stratum is a census when fully judged -> no sampling variance there.
    v_h = 0.0 if len(jh) >= n_hit else W_h ** 2 * p_h * (1 - p_h) / max(len(jh) - 1, 1)
    v_n = (W_n ** 2 * p_n * (1 - p_n) / max(len(jn) - 1, 1) * (1 - len(jn) / n_non)
           if n_non else 0.0)
    return est, math.sqrt(v_h + v_n)


@click.command()
@click.option("--baseline", default=str(ART / "judged.jsonl"))
@click.option("--second", default=str(ART / "judged_claude-sonnet-5.jsonl"))
@click.option("--traces", default=str(ART / "traces_k5.json"))
@click.option("--baseline-name", default="haiku-4.5")
@click.option("--second-name", default="sonnet-5")
def main(baseline, second, traces, baseline_name, second_name):
    A, B = load(baseline), load(second)
    shared = sorted(set(A) & set(B))
    print(f"{baseline_name}: {len(A)} parsed   {second_name}: {len(B)} parsed   "
          f"matched: {len(shared)}")
    if not shared:
        raise SystemExit("no overlapping rows — has the second judge run?")
    only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
    if only_a or only_b:
        print(f"  (unmatched: {only_a} only in {baseline_name}, "
              f"{only_b} only in {second_name})")
    print()

    sizes = stratum_sizes(json.loads(pathlib.Path(traces).read_text()))
    A_shared = {k: A[k] for k in shared}
    B_shared = {k: B[k] for k in shared}

    for variant, flagfn in (("RAW judge output", raw_flags),
                            ("CORRECTED (echo + game-recognition removed)",
                             corrected_flags)):
        print("=" * 74)
        print(variant)
        print("=" * 74)
        for label, idx in FLAGS:
            a = [flagfn(A[k])[idx] for k in shared]
            b = [flagfn(B[k])[idx] for k in shared]
            yy, yn, ny, nn = confusion(a, b)
            n = len(shared)
            agree = (yy + nn) / n
            k = kappa(a, b)
            lo, hi = kappa_ci(a, b)
            p = mcnemar_exact(yn, ny)
            print(f"\n{label}")
            print(f"  prevalence      {baseline_name} {100 * sum(a) / n:5.2f}%  "
                  f"{second_name} {100 * sum(b) / n:5.2f}%")
            print(f"  confusion       both-yes {yy:4d}   only-{baseline_name} {yn:4d}   "
                  f"only-{second_name} {ny:4d}   both-no {nn:5d}")
            print(f"  raw agreement   {100 * agree:.2f}%")
            print(f"  Cohen's kappa   {k:.3f}  [{lo:.3f}, {hi:.3f}]")
            print(f"  pos. specific   {100 * psa(yy, yn, ny):.1f}%   "
                  f"(agreement restricted to the flagged traces)")
            print(f"  McNemar exact   p = {p:.4g}"
                  + ("   <- judges differ systematically, not just noisily"
                     if p < 0.05 else ""))

        # Does the headline survive the judge swap?
        print(f"\n  population estimates by CoT type ({variant.split()[0].lower()})")
        print(f"  {'group':12s} {'flag':26s} {baseline_name:>18s} {second_name:>18s}")
        for gname, members in (("raw CoT", RAW_COT), ("summarized", SUMMARIZED)):
            for label, idx in FLAGS:
                ea, sa = population_estimate(A_shared, sizes[gname],
                                             members, idx, flagfn)
                eb, sb = population_estimate(B_shared, sizes[gname],
                                             members, idx, flagfn)
                print(f"  {gname:12s} {label:26s} "
                      f"{100 * ea:6.2f}% +-{100 * 1.96 * sa:4.2f} "
                      f"{100 * eb:6.2f}% +-{100 * 1.96 * sb:4.2f}")
        print()

    # ---- per-model agreement (corrected) -----------------------------------
    print("=" * 74)
    print("per-model agreement (corrected flags)")
    print("=" * 74)
    print(f"{'model':24s} {'CoT':>11s} {'n':>5s} "
          f"{'aware ' + baseline_name:>16s} {'aware ' + second_name:>16s} "
          f"{'agree':>7s} {'kappa':>7s}")
    by = defaultdict(list)
    for kk in shared:
        by[A[kk]["model"]].append(kk)
    for m in sorted(by, key=lambda x: (cot_kind(x), x)):
        ks = by[m]
        a = [corrected_flags(A[k])[0] for k in ks]
        b = [corrected_flags(B[k])[0] for k in ks]
        agree = sum(x == y for x, y in zip(a, b)) / len(ks)
        kap = kappa(a, b)
        print(f"{m:24s} {cot_kind(m):>11s} {len(ks):5d} "
              f"{100 * sum(a) / len(ks):15.2f}% {100 * sum(b) / len(ks):15.2f}% "
              f"{100 * agree:6.2f}% {kap:7.3f}")

    # ---- per-stratum -------------------------------------------------------
    print(f"\nper-stratum agreement (corrected, awareness)")
    for strat in ("prefilter_hit", "random_nonhit"):
        ks = [k for k in shared if A[k].get("stratum") == strat]
        if not ks:
            continue
        a = [corrected_flags(A[k])[0] for k in ks]
        b = [corrected_flags(B[k])[0] for k in ks]
        print(f"  {strat:14s} n={len(ks):5d}  "
              f"{baseline_name} {100 * sum(a) / len(ks):5.2f}%  "
              f"{second_name} {100 * sum(b) / len(ks):5.2f}%  "
              f"agree {100 * sum(x == y for x, y in zip(a, b)) / len(ks):.2f}%  "
              f"kappa {kappa(a, b):.3f}")

    print("\nReading this: high raw agreement at a ~3% base rate is mostly the two")
    print("judges agreeing that nothing is happening. Kappa and positive specific")
    print("agreement are the numbers that speak to whether the FLAGS are real.")


if __name__ == "__main__":
    main()
