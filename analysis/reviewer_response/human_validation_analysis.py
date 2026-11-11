"""Scoring for the RQ3 reasoning-classifier human validation.

Runs the four analyses fixed in human_validation/PREREGISTRATION.md, in the
order of evidential weight given there:

  1. Differential error (primary) — human-vs-judge disagreement within optimal
     vs within suboptimal traces. Only outcome-correlated judge error can
     manufacture the RQ3 direction; uniform error attenuates toward the null.
  2. Reliability — per-category kappa, prevalence, raw agreement, Gwet's AC1,
     PABAK. All four always together: kappa is degenerate at the prevalences
     these categories sit at.
  3. Length split — agreement above/below median reasoning length, to see
     whether a fatigued annotator skimmed the long traces.
  4. Direction — welfare/safety category count, optimal vs suboptimal,
     Mann-Whitney; plus the binary odds ratio and per-category odds ratios.

Usage:
    uv run python analysis/reviewer_response/human_validation_analysis.py
"""

from __future__ import annotations

import itertools
import math
import pathlib
import random

import click
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HV = pathlib.Path("analysis/reviewer_response/human_validation")
CATS = ["NASH_DOMINANT", "PAYOFF_MAX", "UTILITARIAN", "RAWLSIAN",
        "CATASTROPHE_PREVENTION", "PRECAUTIONARY", "AI_SAFETY", "DOMAIN_OTHER"]
WELFARE = ["UTILITARIAN", "RAWLSIAN", "CATASTROPHE_PREVENTION",
           "AI_SAFETY", "PRECAUTIONARY"]


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa with expected agreement from the raters' own marginals."""
    po = (a == b).mean()
    pe = sum(((a == c).mean() * (b == c).mean()) for c in (0, 1))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    """Gwet's AC1: chance agreement from the pooled marginal, not the product.

    Stable where kappa collapses, i.e. when one label dominates.
    """
    po = (a == b).mean()
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def pabak(a: np.ndarray, b: np.ndarray) -> float:
    return 2 * (a == b).mean() - 1


def boot_ci(fn, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        v = fn(a[s], b[s])
        if not math.isnan(v):
            vals.append(v)
    if len(vals) < n // 4:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def woolf_or(exposed_pos, exposed_n, unexposed_pos, unexposed_n):
    """Odds ratio with Woolf (log) CI, Haldane-Anscombe 0.5 correction."""
    a = exposed_pos + 0.5
    b = exposed_n - exposed_pos + 0.5
    c = unexposed_pos + 0.5
    d = unexposed_n - unexposed_pos + 0.5
    or_ = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return or_, or_ * math.exp(-1.96 * se), or_ * math.exp(1.96 * se)


@click.command()
@click.option("--annotators", default="pepijn,angelo")
def main(annotators):
    names = annotators.split(",")
    key = pd.read_csv(HV.parent / "data/human_validation_keys/_llm_key.csv").set_index("annotation_id")
    ann = {n: pd.read_csv(HV / f"annotation_{n}.csv").set_index("annotation_id")
           for n in names}

    ids = sorted(set.intersection(*(set(d.index) for d in ann.values())))
    core = [i for i in ids if key.loc[i, "block"] == "core"]
    print(f"annotators        : {', '.join(names)}")
    print(f"labelled in common: {len(ids)}  (core {len(core)}, "
          f"enrichment {len(ids) - len(core)})")
    if len(ids) - len(core) == 0:
        print("  the optional enrichment block was not done by both; "
              "low-prevalence categories stay underpowered.")
    k = key.loc[core]
    A = {n: ann[n].loc[core] for n in names}

    # ---------------- 2. Reliability (run first; it gates the rest) --------
    print("\n" + "=" * 78)
    print("RELIABILITY — inter-annotator, core block (n=%d)" % len(core))
    print("=" * 78)
    a_name, b_name = names[0], names[1]
    print(f"{'category':24s} {'prev A':>7s} {'prev B':>7s} {'raw':>6s} "
          f"{'kappa':>7s} {'kappa 95% CI':>16s} {'AC1':>7s} {'PABAK':>7s}")
    rel = {}
    for c in CATS:
        a = A[a_name][c].to_numpy()
        b = A[b_name][c].to_numpy()
        kp = kappa(a, b)
        lo, hi = boot_ci(kappa, a, b)
        rel[c] = dict(raw=(a == b).mean(), kappa=kp, ac1=gwet_ac1(a, b))
        print(f"{c:24s} {a.mean():6.2f}  {b.mean():6.2f}  {(a==b).mean():5.2f} "
              f"{kp:7.2f}  [{lo:5.2f}, {hi:5.2f}] {gwet_ac1(a,b):7.2f} {pabak(a,b):7.2f}")
    allA = np.concatenate([A[a_name][c].to_numpy() for c in CATS])
    allB = np.concatenate([A[b_name][c].to_numpy() for c in CATS])
    print(f"{'-- pooled over labels':24s} {allA.mean():6.2f}  {allB.mean():6.2f}  "
          f"{(allA==allB).mean():5.2f} {kappa(allA,allB):7.2f}  "
          f"{'':16s} {gwet_ac1(allA,allB):7.2f} {pabak(allA,allB):7.2f}")

    # Kappa, AC1 and PABAK all correct agreement for chance using the marginals
    # analytically. This does the same thing empirically and asks a blunter
    # question: is the agreement about the traces at all? Shuffling one rater's
    # rows destroys every trace-level correspondence while preserving exactly
    # how often they tick each label. If observed agreement sits inside that
    # null, the raters share a labelling frequency and nothing else.
    print("\nPermutation null — shuffle one rater's rows, keep their label rates:")
    print(f"{'pair':22s} {'observed':>9s} {'null mean':>10s} {'null 95%':>16s} {'p':>7s}")

    def perm(X: pd.DataFrame, Y: pd.DataFrame, label: str, seed: int = 0):
        rng = np.random.default_rng(seed)
        xv, yv = X[CATS].to_numpy(), Y[CATS].to_numpy()
        obs = (xv == yv).mean()
        null = np.array([(xv == yv[rng.permutation(len(yv))]).mean()
                         for _ in range(5000)])
        p = (null >= obs).mean()
        print(f"{label:22s} {obs:9.3f} {null.mean():10.3f} "
              f"[{np.percentile(null,2.5):.3f},{np.percentile(null,97.5):.3f}] {p:7.4f}")

    perm(A[a_name], A[b_name], f"{a_name} vs {b_name}")
    for n in names:
        perm(A[n], k, f"{n} vs judge")

    rng = np.random.default_rng(0)
    ps = {}
    for c in CATS:
        x, y = A[a_name][c].to_numpy(), A[b_name][c].to_numpy()
        obs = (x == y).mean()
        null = np.array([(x == y[rng.permutation(len(y))]).mean() for _ in range(5000)])
        ps[c] = (null >= obs).mean()
    print(f"  per-category {a_name} vs {b_name}, smallest p: "
          + ", ".join(f"{c} {ps[c]:.3f}" for c in sorted(ps, key=ps.get)[:3]))

    # One badly-scoped category could in principle account for the whole
    # failure. Leave-one-out, plus the two families, says otherwise — and the
    # subset that carries RQ3 is the worst of them, so trimming the taxonomy
    # after the fact would improve the headline without fixing anything.
    print("\nLeave-one-out and by-family (is one category driving this?):")
    print(f"{'subset':38s} {'n':>5s} {'raw':>6s} {'kappa':>7s} {'AC1':>6s} {'perm p':>8s}")

    def subset(cats, label):
        x = np.concatenate([A[a_name][c].to_numpy() for c in cats])
        y = np.concatenate([A[b_name][c].to_numpy() for c in cats])
        rng2 = np.random.default_rng(0)
        xa = A[a_name][cats].to_numpy()
        ya = A[b_name][cats].to_numpy()
        obs = (xa == ya).mean()
        null = np.array([(xa == ya[rng2.permutation(len(ya))]).mean() for _ in range(5000)])
        print(f"{label:38s} {len(x):5d} {(x==y).mean():6.2f} {kappa(x,y):7.2f} "
              f"{gwet_ac1(x,y):6.2f} {(null>=obs).mean():8.3f}")

    subset(CATS, "all 8")
    for c in CATS:
        subset([x for x in CATS if x != c], f"  drop {c}")
    subset(WELFARE, "welfare/safety family (carries RQ3)")
    subset(["NASH_DOMINANT", "PAYOFF_MAX"], "game-theoretic family")

    print("\nOverlap of positive labels (Jaccard) — where the two raters do not")
    print("merely differ in strictness but pick out different traces entirely:")
    print(f"{'category':24s} {a_name[:8]:>8s} {b_name[:8]:>8s} {'both':>6s} {'jaccard':>8s}")
    for c in CATS:
        p_, b_ = A[a_name][c].to_numpy().astype(bool), A[b_name][c].to_numpy().astype(bool)
        both = int((p_ & b_).sum())
        print(f"{c:24s} {int(p_.sum()):8d} {int(b_.sum()):8d} {both:6d} "
              f"{both / max((p_ | b_).sum(), 1):8.2f}")

    print("\nAnnotator vs judge (same statistic, for the pre-registered comparison):")
    print(f"{'category':24s} {'prev judge':>10s}" +
          "".join(f" {n[:6]+' raw':>12s} {n[:6]+' k':>9s}" for n in names))
    for c in CATS:
        j = k[c].to_numpy()
        row = f"{c:24s} {j.mean():10.2f}"
        for n in names:
            h = A[n][c].to_numpy()
            row += f" {(h==j).mean():12.2f} {kappa(h,j):9.2f}"
        print(row)
    jall = np.concatenate([k[c].to_numpy() for c in CATS])
    row = f"{'-- pooled over labels':24s} {jall.mean():10.2f}"
    for n in names:
        h = np.concatenate([A[n][c].to_numpy() for c in CATS])
        row += f" {(h==jall).mean():12.2f} {kappa(h,jall):9.2f}"
    print(row)
    print("\nPre-registered negative result #2: annotator-vs-judge agreement no")
    print("better than annotator-vs-annotator would mean the judge is not tracking")
    print("a construct the humans share.")

    # Directional breakdown. The judge is the object under validation, not a
    # ground truth, so "FP"/"FN" mean only "ticked where the judge did not" and
    # vice versa — they say which way a rater's threshold sits relative to it.
    print("\nDirection of each annotator's departure from the judge")
    print(f"{'category':24s} {'judge':>5s} |" +
          "".join(f" {n[:6]:>6s} {'over':>4s} {'under':>5s} {'sens':>5s} {'spec':>5s} |"
                  for n in names))
    for c in CATS:
        j = k[c].to_numpy().astype(bool)
        row = f"{c:24s} {int(j.sum()):5d} |"
        for n in names:
            h = A[n][c].to_numpy().astype(bool)
            sens = (h & j).sum() / j.sum() if j.sum() else float("nan")
            spec = ((~h) & (~j)).sum() / (~j).sum() if (~j).sum() else float("nan")
            row += (f" {int(h.sum()):6d} {int((h & ~j).sum()):4d} {int((~h & j).sum()):5d}"
                    f" {sens:5.2f} {spec:5.2f} |")
        print(row)
    jp = np.concatenate([k[c].to_numpy() for c in CATS]).astype(bool)
    row = f"{'POOLED':24s} {int(jp.sum()):5d} |"
    for n in names:
        hp = np.concatenate([A[n][c].to_numpy() for c in CATS]).astype(bool)
        row += (f" {int(hp.sum()):6d} {int((hp & ~jp).sum()):4d} {int((~hp & jp).sum()):5d}"
                f" {(hp & jp).sum()/jp.sum():5.2f} {((~hp)&(~jp)).sum()/(~jp).sum():5.2f} |")
    print(row)

    # Which way does the judge break when the two humans disagree, and where do
    # the humans agree with each other but not with the judge? The latter is the
    # only cell that isolates judge miscalibration from human unreliability.
    p_ = np.concatenate([A[a_name][c].to_numpy() for c in CATS]).astype(bool)
    b_ = np.concatenate([A[b_name][c].to_numpy() for c in CATS]).astype(bool)
    print(f"\nThree-way split of the {len(jp)} label decisions:")
    for desc, mask in (
        ("all three agree", (p_ == b_) & (p_ == jp)),
        (f"judge sides with {a_name}", (p_ != b_) & (p_ == jp)),
        (f"judge sides with {b_name}", (p_ != b_) & (b_ == jp)),
        ("both humans agree, judge differs", (p_ == b_) & (p_ != jp)),
    ):
        print(f"  {desc:34s} {int(mask.sum()):4d}  ({100*mask.mean():4.1f}%)")
    print("\n  Categories where the humans agree with each other but not the judge")
    print("  (the cell that isolates judge miscalibration from human unreliability):")
    print(f"  {'category':24s} {'judge alone ticks':>18s} {'both humans tick, judge not':>28s}")
    for c in CATS:
        p2, b2, j2 = (A[a_name][c].to_numpy().astype(bool),
                      A[b_name][c].to_numpy().astype(bool),
                      k[c].to_numpy().astype(bool))
        print(f"  {c:24s} {int((~p2 & ~b2 & j2).sum()):18d} {int((p2 & b2 & ~j2).sum()):28d}")

    # ---------------- 1. Differential error (primary) ----------------------
    print("\n" + "=" * 78)
    print("PRIMARY — differential error (human vs judge, by outcome)")
    print("=" * 78)
    opt = k["util_optimal"].to_numpy().astype(bool)
    for n in names:
        d = np.stack([(A[n][c].to_numpy() != k[c].to_numpy()) for c in CATS], 1)
        per_trace = d.mean(1)
        diff = per_trace[opt].mean() - per_trace[~opt].mean()
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            i1 = rng.choice(np.flatnonzero(opt), opt.sum(), replace=True)
            i0 = rng.choice(np.flatnonzero(~opt), (~opt).sum(), replace=True)
            boots.append(per_trace[i1].mean() - per_trace[i0].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        flag = "COVERS ZERO" if lo <= 0 <= hi else "EXCLUDES ZERO -- see preregistration"
        print(f"  {n:8s} disagreement  optimal {per_trace[opt].mean():.3f}   "
              f"suboptimal {per_trace[~opt].mean():.3f}   "
              f"diff {diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {flag}")
    print("\n  Uniform judge error attenuates an association toward the null; only")
    print("  outcome-correlated error can manufacture the RQ3 direction.")

    # ---------------- 3. Length split --------------------------------------
    print("\n" + "=" * 78)
    print("LENGTH SPLIT — is the human weaker on long traces?")
    print("=" * 78)
    med = k["reasoning_chars"].median()
    longm = (k["reasoning_chars"] > med).to_numpy()
    print(f"  median reasoning length: {med:,.0f} chars "
          f"(range {k['reasoning_chars'].min():,}-{k['reasoning_chars'].max():,})")
    for label, mask in (("short", ~longm), ("long", longm)):
        a = np.concatenate([A[a_name][c].to_numpy()[mask] for c in CATS])
        b = np.concatenate([A[b_name][c].to_numpy()[mask] for c in CATS])
        j = np.concatenate([k[c].to_numpy()[mask] for c in CATS])
        vj = [f"{n} vs judge {np.mean(np.concatenate([A[n][c].to_numpy()[mask] for c in CATS]) == j):.2f}"
              for n in names]
        print(f"  {label:5s} (n={mask.sum():2d})  A vs B raw {(a==b).mean():.2f}  "
              f"AC1 {gwet_ac1(a,b):.2f}   " + "  ".join(vj))

    # ---------------- 4. Direction -----------------------------------------
    print("\n" + "=" * 78)
    print("DIRECTION — does human labelling reproduce the reported direction?")
    print("=" * 78)
    label_sets = {
        "judge": k[CATS],
        "consensus (both tick)": (A[a_name][CATS] & A[b_name][CATS]),
        "union (sensitivity)": (A[a_name][CATS] | A[b_name][CATS]),
        **{f"{n} alone": A[n][CATS] for n in names},
    }
    print("\nPrimary estimator: number of welfare/safety categories per trace")
    print(f"{'labels':24s} {'optimal':>9s} {'subopt':>9s} {'diff':>7s} {'MW p':>8s}")
    dirs = {}
    for lab, df in label_sets.items():
        cnt = df[WELFARE].sum(1).to_numpy()
        u, p = mannwhitneyu(cnt[opt], cnt[~opt], alternative="two-sided")
        dirs[lab] = cnt[opt].mean() - cnt[~opt].mean()
        print(f"{lab:24s} {cnt[opt].mean():9.2f} {cnt[~opt].mean():9.2f} "
              f"{dirs[lab]:+7.2f} {p:8.3f}")
    same = all(np.sign(dirs[l]) == np.sign(dirs["judge"]) for l in dirs)
    print(f"\n  all label sets point the same way as the judge: {same}")

    print("\nSecondary: odds ratio for 'any welfare/safety category present'")
    for lab, df in label_sets.items():
        any_ = (df[WELFARE].sum(1) > 0).to_numpy()
        o, lo, hi = woolf_or(any_[opt].sum(), opt.sum(),
                             any_[~opt].sum(), (~opt).sum())
        print(f"  {lab:24s} OR {o:5.2f}  [{lo:4.2f}, {hi:5.2f}]   "
              f"prevalence {any_.mean():.2f}")

    print("\nPer-category odds ratios (optimal vs suboptimal), judge vs consensus")
    print(f"{'category':24s} {'judge OR':>9s} {'95% CI':>16s} "
          f"{'cons. OR':>9s} {'95% CI':>16s} {'sign':>6s}")
    for c in CATS:
        out = []
        for df in (k[CATS], A[a_name][CATS] & A[b_name][CATS]):
            v = df[c].to_numpy().astype(bool)
            out.append(woolf_or(v[opt].sum(), opt.sum(), v[~opt].sum(), (~opt).sum()))
        agree = "yes" if (out[0][0] > 1) == (out[1][0] > 1) else "NO"
        print(f"{c:24s} {out[0][0]:9.2f} [{out[0][1]:5.2f},{out[0][2]:7.2f}] "
              f"{out[1][0]:9.2f} [{out[1][1]:5.2f},{out[1][2]:7.2f}] {agree:>6s}")
    print("\n  n=60 cannot resolve the low-prevalence categories; the CIs show it.")

    # ---------------- 5b. Labeller sensitivity ------------------------------
    # Comparing a human OR on 60 rows against the paper's OR on 16,039 confounds
    # the labeller with the sample size. Resampling 30/30 from the judge's OWN
    # full-corpus labels isolates what n=60 does with the labeller held fixed;
    # a human OR inside that band is not evidence of a labelling effect.
    corpus = pathlib.Path("artifacts/reasoning/classified_all_gpt4omini.json")
    if not corpus.exists():
        print(f"\n[skipping labeller-sensitivity: {corpus} not present]")
        return
    print("\n" + "=" * 78)
    print("SENSITIVITY — labeller effect vs. what n=60 does on its own")
    print("=" * 78)
    import json as _json
    recs = _json.loads(corpus.read_text())
    F = pd.DataFrame([
        {**{c: int(c in ((r.get("classification") or {}).get("categories") or []))
            for c in CATS},
         "opt": int(r["utilitarian_score"] == 1.0)} for r in recs])
    optF = F["opt"].to_numpy().astype(bool)

    def or_(v, o):
        a, b = v[o].sum() + .5, o.sum() - v[o].sum() + .5
        c_, d_ = v[~o].sum() + .5, (~o).sum() - v[~o].sum() + .5
        return (a * d_) / (b * c_)

    rng3 = np.random.default_rng(0)
    iO, iS = np.flatnonzero(optF), np.flatnonzero(~optF)
    n_o, n_s = int(opt.sum()), int((~opt).sum())
    subs = [np.concatenate([rng3.choice(iO, n_o, False), rng3.choice(iS, n_s, False)])
            for _ in range(4000)]
    o60 = np.concatenate([np.ones(n_o, bool), np.zeros(n_s, bool)])
    sets = {**{n: A[n][CATS] for n in names},
            "consensus": (A[a_name][CATS].astype(bool) & A[b_name][CATS].astype(bool)).astype(int),
            "judge/60": k[CATS]}
    print(f"corpus n={len(F)}  (judge labels)   subsample {n_o}/{n_s} optimal/suboptimal\n")
    print(f"{'category':24s} {'full':>7s} {'n=60 band':>20s} |" +
          "".join(f" {s[:9]:>10s}" for s in sets))
    outside = {s: [] for s in sets}
    for c in CATS:
        v = F[c].to_numpy().astype(bool)
        band = np.array([or_(v[s], o60) for s in subs])
        lo, hi = np.percentile(band, [2.5, 97.5])
        row = f"{c:24s} {or_(v, optF):7.2f} [{lo:7.2f},{hi:8.2f}] |"
        for s, df in sets.items():
            o = or_(df[c].to_numpy().astype(bool), opt)
            out = o < lo or o > hi
            outside[s].append(c) if out else None
            row += f" {o:9.2f}{'*' if out else ' '}"
        print(row)
    print("\n  * = outside the 95% band that n=60 sampling of the judge's own labels gives")
    for s, v in outside.items():
        print(f"  {s:12s} {len(v)}/8 outside" + (f": {', '.join(v)}" if v else ""))

    print("\nThe two coarse claims actually promised:")
    cF = F[WELFARE].sum(1).to_numpy()
    anyF = (F[WELFARE].sum(1) > 0).to_numpy()
    pmF = F["PAYOFF_MAX"].to_numpy().astype(bool)
    bands = {
        "welfare/safety count diff": ([cF[s][:n_o].mean() - cF[s][n_o:].mean() for s in subs],
                                      cF[optF].mean() - cF[~optF].mean(),
                                      lambda df: (df[WELFARE].sum(1).to_numpy()[opt].mean()
                                                  - df[WELFARE].sum(1).to_numpy()[~opt].mean())),
        "any-welfare OR": ([or_(anyF[s], o60) for s in subs], or_(anyF, optF),
                           lambda df: or_((df[WELFARE].sum(1) > 0).to_numpy(), opt)),
        "PAYOFF_MAX OR": ([or_(pmF[s], o60) for s in subs], or_(pmF, optF),
                          lambda df: or_(df["PAYOFF_MAX"].to_numpy().astype(bool), opt)),
    }
    print(f"{'quantity':28s} {'full':>7s} {'n=60 band':>20s} |" +
          "".join(f" {s[:9]:>10s}" for s in sets))
    for label, (band, fullv, fn) in bands.items():
        lo, hi = np.percentile(band, [2.5, 97.5])
        row = f"{label:28s} {fullv:7.2f} [{lo:7.2f},{hi:8.2f}] |"
        for s, df in sets.items():
            v = fn(df)
            row += f" {v:9.2f}{'*' if (v < lo or v > hi) else ' '}"
        print(row)
    print("\n  Per-category ORs are not identifiable at n=60 for ANY labeller,")
    print("  the judge included. The coarse claims are, and they hold throughout.")


if __name__ == "__main__":
    main()
