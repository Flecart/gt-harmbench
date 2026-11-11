"""Score human annotations of reasoning traces against each other and the judge.

    uv run python -m analysis.reviewer_response.score_human_validation \
        human_validation/annotation_a.csv human_validation/annotation_b.csv

    # exercise the whole pipeline before any real labels exist:
    uv run python -m analysis.reviewer_response.score_human_validation --dry-run

Reports, in the order they matter (see human_validation/PREREGISTRATION.md):

  1. DIFFERENTIAL ERROR -- does human-vs-judge disagreement differ between
     optimal and suboptimal traces? This is the check that defends RQ3.
     Uniform judge error attenuates an association toward the null; it cannot
     manufacture one. Only outcome-CORRELATED judge error can. If the two
     disagreement rates match, the direction cannot be an artifact of the
     judge, and that holds even if kappa is middling and the CIs are wide.
  2. RELIABILITY -- inter-annotator agreement, and each annotator vs the judge,
     per category, with prevalence, bootstrap CIs, and prevalence-robust
     coefficients alongside plain kappa.
  3. LENGTH SPLIT -- the same agreement, short vs long traces. The judge read
     every word; a tired human may not have. This says whether it matters.
  4. DIRECTION -- the two claims actually promised on OpenReview, plus the full
     per-category odds-ratio table, computed under human labels and under judge
     labels on the SAME rows.

CORE rows only for anything directional. The enrichment block is deliberately
not outcome-representative, so it is used for agreement only.
"""
from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

import numpy as np

HV = Path(__file__).resolve().parent / "human_validation"
CATS = ["NASH_DOMINANT", "PAYOFF_MAX", "UTILITARIAN", "RAWLSIAN",
        "CATASTROPHE_PREVENTION", "PRECAUTIONARY", "AI_SAFETY", "DOMAIN_OTHER"]
# Pre-registered: the "welfare and safety oriented" family whose prevalence the
# OpenReview reply claims is higher under optimal decisions.
WELFARE = ["UTILITARIAN", "RAWLSIAN", "CATASTROPHE_PREVENTION", "AI_SAFETY",
           "PRECAUTIONARY"]
BOOT = 2000
SEED = 42


# ---------------------------------------------------------------- loading

def load_labels(path) -> dict:
    return {int(r["annotation_id"]): {c: int(r.get(c) or 0) for c in CATS}
            for r in csv.DictReader(open(path))}


def load_meta() -> dict:
    return {int(r["annotation_id"]): {"block": r["block"],
                                      "opt": int(r["util_optimal"]),
                                      "chars": int(r["reasoning_chars"]),
                                      "game": r["formal_game"]}
            for r in csv.DictReader(open(HV.parent / "data/human_validation_keys/_llm_key.csv"))}


# ---------------------------------------------------------------- statistics

def kappa(a, b) -> float:
    """Cohen's kappa, binary. Expected agreement from the raters' own marginals
    -- NOT a hardcoded constant (that bug is what made the earlier quality-rating
    figure wrong)."""
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan")


def ac1(a, b) -> float:
    """Gwet's AC1. Same numerator as kappa, but expected agreement is derived
    from the mean positive rate rather than the product of marginals, so it does
    not collapse when a category is near-universal or near-absent. Reported
    alongside kappa because NASH_DOMINANT sits at ~95% prevalence, where kappa
    can be near zero despite excellent raw agreement."""
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan")


def pabak(a, b) -> float:
    """Prevalence-adjusted bias-adjusted kappa = 2*po - 1."""
    return 2 * np.mean(np.asarray(a) == np.asarray(b)) - 1


def boot_ci(fn, a, b, rng, n=BOOT):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        v = fn(a[idx], b[idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def odds_ratio(pos_opt, n_opt, pos_sub, n_sub):
    """Haldane-Anscombe corrected OR with a Woolf 95% CI."""
    a, b = pos_opt + 0.5, n_opt - pos_opt + 0.5
    c, d = pos_sub + 0.5, n_sub - pos_sub + 0.5
    orv = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orv, math.exp(math.log(orv) - 1.96 * se), math.exp(math.log(orv) + 1.96 * se)


def mannwhitney_p(x, y) -> float:
    """Two-sided Mann-Whitney U with a normal approximation and tie correction.
    Used for the pre-registered primary estimator (welfare-category COUNT per
    trace), which the binarised 'any welfare category' version loses power to
    because it hits a ceiling."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    allv = np.concatenate([x, y])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for g in np.where(cnt > 1)[0]:
        m = inv == g
        ranks[m] = ranks[m].mean()
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    tie = sum(c ** 3 - c for c in cnt)
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1)))) if n > 1 else 0
    if sd == 0:
        return float("nan")
    z = (u1 - mu) / sd
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


# ---------------------------------------------------------------- reports

def report_differential_error(human, judge, meta, ids):
    print("\n" + "=" * 72)
    print("1. DIFFERENTIAL ERROR  (the check that defends RQ3)")
    print("=" * 72)
    o = [i for i in ids if meta[i]["opt"] == 1]
    s = [i for i in ids if meta[i]["opt"] == 0]
    if not o or not s:
        print("  not computable (need both outcome groups)")
        return

    def rate(sub):
        return np.mean([sum(human[i][c] != judge[i][c] for c in CATS) / len(CATS)
                        for i in sub])

    ro, rs = rate(o), rate(s)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(BOOT):
        diffs.append(rate([o[k] for k in rng.integers(0, len(o), len(o))])
                     - rate([s[k] for k in rng.integers(0, len(s), len(s))]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  human-vs-judge per-label disagreement")
    print(f"    optimal traces    {ro:6.3f}   (n={len(o)})")
    print(f"    suboptimal traces {rs:6.3f}   (n={len(s)})")
    print(f"    difference        {ro-rs:+6.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    if lo <= 0 <= hi:
        print("  -> CI includes zero: judge error is outcome-independent, so it")
        print("     cannot have manufactured the reported direction.")
    else:
        print("  -> CI EXCLUDES zero: judge error is outcome-correlated. The")
        print("     reported direction may be partly an artifact. Investigate")
        print("     per-category before reporting.")

    print("\n  per-category, judge minus human positive rate:")
    print(f"  {'category':24s}{'optimal':>10}{'subopt':>10}{'gap diff':>10}")
    for c in CATS:
        bo = np.mean([judge[i][c] - human[i][c] for i in o])
        bs = np.mean([judge[i][c] - human[i][c] for i in s])
        print(f"  {c:24s}{bo:+10.3f}{bs:+10.3f}{bo-bs:+10.3f}")


def report_agreement(name, x, y, ids, meta):
    print("\n" + "=" * 72)
    print(f"2. AGREEMENT: {name}   (n={len(ids)})")
    print("=" * 72)
    rng = np.random.default_rng(SEED)
    print(f"  {'category':24s}{'prev':>6}{'raw%':>7}{'kappa':>8}"
          f"{'95% CI':>16}{'AC1':>7}{'PABAK':>7}")
    ks = []
    for c in CATS:
        v1 = np.array([x[i][c] for i in ids])
        v2 = np.array([y[i][c] for i in ids])
        k = kappa(v1, v2)
        lo, hi = boot_ci(kappa, v1, v2, rng)
        ks.append(k)
        prev = (v1.mean() + v2.mean()) / 2
        print(f"  {c:24s}{prev:6.2f}{100*np.mean(v1==v2):7.1f}{k:8.3f}"
              f"   [{lo:5.2f},{hi:5.2f}]{ac1(v1,v2):7.3f}{pabak(v1,v2):7.3f}")
    exact = np.mean([all(x[i][c] == y[i][c] for c in CATS) for i in ids])
    print(f"  {'macro (mean of above)':24s}{'':6}{'':7}{np.nanmean(ks):8.3f}")
    print(f"  exact set match: {100*exact:.1f}% of traces")
    if np.isnan(ks).any():
        print("  NOTE: kappa undefined for "
              f"{[c for c,k in zip(CATS,ks) if math.isnan(k)]} "
              "(no variation); excluded from the macro average.")


def report_length_split(x, y, ids, meta, name):
    med = float(np.median([meta[i]["chars"] for i in ids]))
    short = [i for i in ids if meta[i]["chars"] <= med]
    long_ = [i for i in ids if meta[i]["chars"] > med]
    print("\n" + "=" * 72)
    print(f"3. LENGTH SPLIT: {name}   (median {med:,.0f} chars)")
    print("=" * 72)
    print(f"  {'category':24s}{'short kappa':>13}{'long kappa':>13}")
    for c in CATS:
        ss = kappa([x[i][c] for i in short], [y[i][c] for i in short])
        ll = kappa([x[i][c] for i in long_], [y[i][c] for i in long_])
        print(f"  {c:24s}{ss:13.3f}{ll:13.3f}")
    raw_s = np.mean([np.mean([x[i][c] == y[i][c] for c in CATS]) for i in short])
    raw_l = np.mean([np.mean([x[i][c] == y[i][c] for c in CATS]) for i in long_])
    print(f"  raw agreement: short {100*raw_s:.1f}%  (n={len(short)})   "
          f"long {100*raw_l:.1f}%  (n={len(long_)})")
    print("  Similar values mean long traces were not skimmed. If long is much")
    print("  worse, the human is the weaker rater there -- the judge read it all.")


def report_direction(name, labels, meta, ids):
    o = [i for i in ids if meta[i]["opt"] == 1]
    s = [i for i in ids if meta[i]["opt"] == 0]
    if not o or not s:
        print(f"\n  [{name}] not computable")
        return
    wo = [sum(labels[i][c] for c in WELFARE) for i in o]
    ws = [sum(labels[i][c] for c in WELFARE) for i in s]
    print(f"\n  --- {name} (opt={len(o)}, subopt={len(s)}) ---")
    print(f"  PRIMARY  welfare/safety categories per trace: "
          f"optimal {np.mean(wo):.2f} vs suboptimal {np.mean(ws):.2f}   "
          f"(Mann-Whitney p={mannwhitney_p(wo, ws):.3f})")
    anyo = sum(1 for v in wo if v > 0)
    anys = sum(1 for v in ws if v > 0)
    orv, lo, hi = odds_ratio(anyo, len(o), anys, len(s))
    print(f"  SECONDARY  any welfare/safety category: "
          f"{anyo}/{len(o)} vs {anys}/{len(s)}   OR {orv:.2f} [{lo:.2f}, {hi:.2f}]")
    print(f"\n  {'category':24s}{'opt%':>7}{'sub%':>7}{'OR':>8}{'95% CI':>18}")
    for c in CATS:
        po = np.mean([labels[i][c] for i in o])
        ps = np.mean([labels[i][c] for i in s])
        orv, lo, hi = odds_ratio(int(sum(labels[i][c] for i in o)), len(o),
                                 int(sum(labels[i][c] for i in s)), len(s))
        print(f"  {c:24s}{100*po:7.1f}{100*ps:7.1f}{orv:8.2f}  [{lo:5.2f},{hi:7.2f}]")


def report_concordance(human, judge, meta, ids):
    print("\n" + "=" * 72)
    print("4. DIRECTION  (core rows only)")
    print("=" * 72)
    report_direction("human consensus", human, meta, ids)
    report_direction("LLM judge, same rows", judge, meta, ids)
    o = [i for i in ids if meta[i]["opt"] == 1]
    s = [i for i in ids if meta[i]["opt"] == 0]
    print("\n  sign concordance (pre-registered definition of 'reproduces'):")
    agree = 0
    for c in CATS:
        hs = odds_ratio(int(sum(human[i][c] for i in o)), len(o),
                        int(sum(human[i][c] for i in s)), len(s))[0]
        js = odds_ratio(int(sum(judge[i][c] for i in o)), len(o),
                        int(sum(judge[i][c] for i in s)), len(s))[0]
        same = (hs > 1) == (js > 1)
        agree += same
        print(f"    {c:24s} human {hs:6.2f}  judge {js:6.2f}  "
              f"{'agree' if same else 'DIFFER'}")
    print(f"    -> {agree}/{len(CATS)} categories agree in sign")


# ---------------------------------------------------------------- main

def synthetic(judge, rng, flip):
    """A fake annotator: the judge's labels with independent flip noise. Lets
    the pipeline be exercised end-to-end before real labels exist. The noise is
    outcome-INDEPENDENT by construction, so a correct implementation must report
    a differential-error CI covering zero."""
    return {i: {c: (1 - v if rng.random() < flip else v)
                for c, v in row.items()} for i, row in judge.items()}


def main() -> None:
    meta = load_meta()
    judge = load_labels(HV.parent / "data/human_validation_keys/_llm_key.csv")

    if "--dry-run" in sys.argv:
        rng = random.Random(SEED)
        A = synthetic(judge, rng, 0.12)
        B = synthetic(judge, rng, 0.18)
        print("DRY RUN -- synthetic annotators (12% and 18% label noise on the")
        print("judge's own labels). Numbers are meaningless; this only checks")
        print("that every statistic computes and the report renders.")
    elif len(sys.argv) >= 3:
        A, B = load_labels(sys.argv[1]), load_labels(sys.argv[2])
    else:
        print(__doc__)
        raise SystemExit(1)

    ids = sorted(set(A) & set(B) & set(judge))
    core = [i for i in ids if meta[i]["block"] == "core"]
    extra = [i for i in ids if meta[i]["block"] == "enrichment"]
    print(f"\nannotated: {len(ids)} traces ({len(core)} core, {len(extra)} enrichment)")
    if len(core) < 60:
        print(f"WARNING: core block incomplete ({len(core)}/60) -- the "
              "directional results below are on a partial sample.")

    # Consensus: a category counts as present when both annotators tick it.
    # Pre-registered as the primary; union is reported as a sensitivity check
    # because it inflates prevalence.
    both = {i: {c: int(A[i][c] and B[i][c]) for c in CATS} for i in ids}
    union = {i: {c: int(A[i][c] or B[i][c]) for c in CATS} for i in ids}

    report_differential_error(both, judge, meta, core)
    report_agreement("annotator A vs B (is the task well defined?)", A, B, ids, meta)
    report_agreement("annotator A vs judge", A, judge, ids, meta)
    report_agreement("annotator B vs judge", B, judge, ids, meta)
    if extra:
        report_agreement("A vs judge, enrichment block only (rare categories)",
                         A, judge, extra, meta)
    report_length_split(A, B, ids, meta, "annotator A vs B")
    report_concordance(both, judge, meta, core)

    print("\n" + "=" * 72)
    print("SENSITIVITY: union consensus instead of intersection")
    print("=" * 72)
    report_direction("human consensus (union)", union, meta, core)


if __name__ == "__main__":
    main()
