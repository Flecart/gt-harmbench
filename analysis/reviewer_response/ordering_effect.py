"""RQ2 order effect (fig:ordering): Coordination with payoffs shown, original
option order (logs/RQ2 - gamified) vs randomly permuted order
(logs/RQ2-coordination), on the 180 Coordination scenarios of the 1,535 release.

Answers are re-parsed (extract_choices) and scored by the chosen option *names*
against the release's utilitarian target, so the permutation cannot matter and
Sonnet's '**ANSWER:**' format (which broke the stored scores) is handled.
Unparsed answers count as wrong (denominator: the 180 scenarios), as in Table 2;
the matched variant scores each model only on scenarios parsed in both orders.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.ordering_effect
"""
from __future__ import annotations

import ast, csv, glob, json
from pathlib import Path

from inspect_ai.log import read_eval_log

from analysis.reviewer_response.extract_choices import ANS_RE, as_actions, map_choice
from analysis.reviewer_response.gamified_comparison import GAMIFIED
from analysis.reviewer_response.reproduce_tables import CLEAN

HERE = Path(__file__).resolve().parent
RANDOM = {  # paper label -> random-order log (logs/RQ2-coordination, Drive "RQ2 - coordination")
    "Claude 4.5 Sonnet": "claude-4.5-sonnet_ZyLM", "GPT-5.2": "gpt-5.2_", "GPT-5.1": "gpt-5.1_",
    "GPT-5 Nano": "gpt-5-nano", "GPT-4o": "gpt-4o-2024", "Grok 4.1 Fast": "grok-4.1-fast",
    "Gemini 3 Flash": "gemini-3-flash", "Gemini 3 Pro": "gemini-3-pro", "Llama 3.3 70B": "llama-3.3-70b",
    "Llama 3.2 3B": "llama-3.2-3b", "Qwen3 30B": "qwen3-30b",
}
def key(story):  # payoff lines are inserted after the first paragraph, so join on it
    return story.split("\n", 1)[0].strip()


def release_coordination():
    rows = [r for r in csv.DictReader(open(CLEAN)) if r["formal_game"] == "Coordination"]
    by_prefix = {key(r["story_row"]): r for r in rows}
    assert len(by_prefix) == len(rows)
    return rows, by_prefix


def targets(r):
    return {tuple(x.lower() for x in ast.literal_eval(t.strip()))
            for t in r["target_utility_maximizing"].split("|")}


def outcomes(path, by_prefix):
    """{release id: True/False correct, or None if unparsed}"""
    res = {}
    for s in read_eval_log(str(path)).samples:
        md = s.metadata or {}
        r = by_prefix.get(key(md.get("story_row") or ""))
        if r is None or r["id"] in res or not s.scores:
            continue
        res[r["id"]] = None
        m = ANS_RE.search(getattr(list(s.scores.values())[0], "answer", None) or "")
        if not m:
            continue
        ar, ac = as_actions(md["actions_row"]), as_actions(md["actions_column"])
        ri, ci = map_choice(m.group(1), ar), map_choice(m.group(2), ac)
        if ri is not None and ci is not None:
            res[r["id"]] = (ar[ri], ac[ci]) in targets(r)
    return res


def main():
    rows, by_prefix = release_coordination()
    n = len(rows)
    out = {}
    print(f"{'model':18s} {'orig':>6s} {'rand':>6s} {'drop':>7s} | {'matched orig':>12s} {'rand':>6s} {'drop':>7s} (n)")
    for m, pat in RANDOM.items():
        rp = [p for p in glob.glob("logs/RQ2-coordination/*.eval") if pat in p]
        if m not in GAMIFIED or len(rp) != 1:
            print(f"{m:18s} missing log"); continue
        o0, o1 = outcomes(GAMIFIED[m], by_prefix), outcomes(rp[0], by_prefix)
        a0 = sum(o0.get(r["id"]) is True for r in rows) / n
        a1 = sum(o1.get(r["id"]) is True for r in rows) / n
        both = [i for i in o0 if o0[i] is not None and o1.get(i) is not None]
        b0 = sum(o0[i] for i in both) / len(both); b1 = sum(o1[i] for i in both) / len(both)
        out[m] = {"original": a0, "random": a1, "matched_original": b0, "matched_random": b1, "matched_n": len(both),
                  "joined": [len(o0), len(o1)]}
        print(f"{m:18s} {a0:6.3f} {a1:6.3f} {a1-a0:+7.3f} | {b0:12.3f} {b1:6.3f} {b1-b0:+7.3f} ({len(both)})  joined {len(o0)}/{len(o1)}")
    for k0, k1, lab in [("original", "random", "fail-wrong"), ("matched_original", "matched_random", "matched")]:
        d = [v[k1] - v[k0] for v in out.values()]
        print(f"mean over {len(d)} ({lab}): original {sum(v[k0] for v in out.values())/len(d):.3f}  "
              f"random {sum(v[k1] for v in out.values())/len(d):.3f}  drop {sum(d)/len(d):+.3f}")
    (HERE / "data" / "ordering_effect.json").write_text(json.dumps(out, indent=1))
    plot(out)


def plot(out):
    """Paper figure (fig:ordering), same style as the submitted ordering.png; matched numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(out)
    o = [out[m]["matched_original"] for m in models]
    r = [out[m]["matched_random"] for m in models]
    fig, ax = plt.subplots(figsize=(20, 9.7))
    ax.set_facecolor("#f0f0f5")
    x = range(len(models))
    ax.bar(x, r, 0.5, color="#5fa5d9", label="Random Order", zorder=3)
    ax.bar(x, [a - b for a, b in zip(o, r)], 0.5, bottom=r, color="#dc5a5a", label="Performance Drop (Delta)", zorder=3)
    for i, (a, b) in enumerate(zip(o, r)):
        ax.text(i, a + 0.045, f"{a:.2f}", ha="center", fontsize=20, fontweight="bold")
        ax.text(i, a + 0.012, f"{b - a:+.3f}", ha="center", fontsize=17, fontweight="bold", color="#d62728")
    ax.set_xticks(list(x)); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score", fontsize=20, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", color="#aaaaaa", zorder=0)
    ax.set_title("Model Performance: Impact of Random Ordering", fontsize=30, fontweight="bold", pad=18)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=18, frameon=True)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "ordering.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
