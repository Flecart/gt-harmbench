"""Inter-judge agreement for the RQ3 reasoning-trace classifier.

Two independent LLM judges (gpt-4o-mini, gpt-5-mini) applied the same multi-label
reasoning taxonomy to the same 16,039 traces. Agreement between them is a
reliability / calibration signal for the LLM-as-judge, to report alongside (and
pending) human validation. Read-only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
J1 = REPO / "artifacts/reasoning/classified_all_gpt4omini.json"
J2 = REPO / "artifacts/reasoning/classified_all_gpt5mini.json"


def trace_key(r):
    # `id` ("0_row") is game+role only, not unique; this composite is unique.
    return (r["game_id"], r["model"], r["player_role"])


def load_labels(path):
    out = {}
    for r in json.load(open(path)):
        cats = (r.get("classification") or {}).get("categories") or []
        out[trace_key(r)] = frozenset(cats)
    return out


def load_labels_and_opt(path):
    """{key: (categories, util_optimal_bool)} for the RQ3 direction check."""
    out = {}
    for r in json.load(open(path)):
        cats = frozenset((r.get("classification") or {}).get("categories") or [])
        opt = r.get("utilitarian_score")
        out[trace_key(r)] = (cats, opt)
    return out


def rq3_direction(path, name, n_boot=2000, seed=0):
    """Per category risk difference RD = P(cat|opt) - P(cat|subopt) with a
    bootstrap 95% CI, on the FULL trace set. This is the threshold-invariant,
    well-powered form of the RQ3 claim (a labeler's liberal/conservative threshold
    shifts both rates together and largely cancels in RD, unlike label kappa)."""
    data = load_labels_and_opt(path)
    cats = sorted({c for v in data.values() for c in v[0]})
    opt = [v[0] for v in data.values() if v[1] == 1.0]
    sub = [v[0] for v in data.values() if v[1] == 0.0]
    rng = np.random.default_rng(seed)
    print(f"\n-- RQ3 direction under judge {name} "
          f"(opt={len(opt)}, subopt={len(sub)}); RD and odds ratio, 95% boot CI --")
    print(f"{'category':26s} {'RD%':>7} {'OR':>6}  {'OR 95% CI':>14}")
    diffs = {}

    def odds(p):
        p = min(max(p, 0.5 / (len(opt) + len(sub)), 1e-6), 1 - 1e-6)
        return p / (1 - p)

    for c in cats:
        po_v = np.array([c in s for s in opt], float)
        ps_v = np.array([c in s for s in sub], float)
        rd = po_v.mean() - ps_v.mean()
        diffs[c] = rd
        bo = rng.integers(0, len(po_v), (n_boot, len(po_v)))
        bs = rng.integers(0, len(ps_v), (n_boot, len(ps_v)))
        or_boot = np.array([odds(po_v[bo[i]].mean()) / odds(ps_v[bs[i]].mean())
                            for i in range(n_boot)])
        orv = odds(po_v.mean()) / odds(ps_v.mean())
        lo, hi = np.percentile(or_boot, [2.5, 97.5])
        print(f"{c:26s} {100*rd:+7.1f} {orv:6.2f}  [{lo:5.2f},{hi:5.2f}]")
    return diffs


def kappa(a, b):
    """Cohen's kappa for two binary label vectors."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    po = np.mean(a == b)
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else float("nan")


def main():
    print("loading judge 1 ..."); l1 = load_labels(J1)
    print("loading judge 2 ..."); l2 = load_labels(J2)
    ids = sorted(set(l1) & set(l2))
    print(f"traces judge1={len(l1)} judge2={len(l2)} common={len(ids)}")

    cats = sorted({c for s in l1.values() for c in s} | {c for s in l2.values() for c in s})
    print(f"\ncategories ({len(cats)}):")
    print(f"{'category':28s} {'j1%':>6} {'j2%':>6} {'raw':>6} {'kappa':>7}")
    kappas = []
    for c in cats:
        v1 = [1 if c in l1[i] else 0 for i in ids]
        v2 = [1 if c in l2[i] else 0 for i in ids]
        k = kappa(v1, v2)
        kappas.append(k)
        raw = np.mean(np.array(v1) == np.array(v2))
        print(f"{c:28s} {100*np.mean(v1):6.1f} {100*np.mean(v2):6.1f} "
              f"{100*raw:6.1f} {k:7.3f}")

    # overall
    exact = np.mean([l1[i] == l2[i] for i in ids])
    jacc = np.mean([len(l1[i] & l2[i]) / len(l1[i] | l2[i]) if (l1[i] | l2[i]) else 1.0
                    for i in ids])
    # micro raw agreement over all (trace, category) cells
    micro = np.mean([[1 if (c in l1[i]) == (c in l2[i]) else 0 for c in cats]
                     for i in ids])
    print("\n=== OVERALL inter-judge agreement ===")
    print(f"mean per-category Cohen kappa (macro): {np.nanmean(kappas):.3f}")
    print(f"micro cell-level raw agreement:        {100*micro:.1f}%")
    print(f"exact label-set match:                 {100*exact:.1f}%")
    print(f"mean Jaccard (set overlap):            {jacc:.3f}")

    # ---- COARSE welfare-vs-payoff agreement (what RQ3 actually uses) ----
    WELFARE = {"UTILITARIAN", "RAWLSIAN", "CATASTROPHE_PREVENTION", "AI_SAFETY",
               "PRECAUTIONARY"}
    PAYOFF = {"PAYOFF_MAX", "NASH_DOMINANT"}
    print("\n=== COARSE agreement (RQ3-relevant buckets) ===")
    print(f"{'bucket':22s} {'j1%':>6} {'j2%':>6} {'raw':>6} {'kappa':>7}")
    for label, bucket in [("has WELFARE/SAFETY", WELFARE), ("has PAYOFF", PAYOFF),
                          ("PAYOFF_MAX only", {"PAYOFF_MAX"})]:
        v1 = [1 if l1[i] & bucket else 0 for i in ids]
        v2 = [1 if l2[i] & bucket else 0 for i in ids]
        raw = np.mean(np.array(v1) == np.array(v2))
        print(f"{label:22s} {100*np.mean(v1):6.1f} {100*np.mean(v2):6.1f} "
              f"{100*raw:6.1f} {kappa(v1, v2):7.3f}")
    # dominant orientation: welfare-leaning vs payoff-leaning (by count of tags)
    def lean(s):
        w, p = len(s & WELFARE), len(s & PAYOFF)
        return "welfare" if w > p else ("payoff" if p > w else "tie")
    dom1 = [lean(l1[i]) for i in ids]; dom2 = [lean(l2[i]) for i in ids]
    print(f"dominant-orientation raw agreement: "
          f"{100*np.mean([a == b for a, b in zip(dom1, dom2)]):.1f}%")

    # ---- does the RQ3 CONCLUSION (direction) survive the judge disagreement? ----
    d1 = rq3_direction(J1, "gpt-4o-mini")
    d2 = rq3_direction(J2, "gpt-5-mini")
    print("\n=== RQ3 direction: cross-judge sign consistency ===")
    print(f"{'category':26s} {'j1 diff':>8} {'j2 diff':>8} {'same sign?':>11}")
    same = 0
    common = sorted(set(d1) & set(d2))
    for c in common:
        s = (d1[c] > 0) == (d2[c] > 0)
        same += s
        print(f"{c:26s} {100*d1[c]:+8.1f} {100*d2[c]:+8.1f} {str(s):>11}")
    print(f"\nsign agreement across judges: {same}/{len(common)} categories")


if __name__ == "__main__":
    main()
