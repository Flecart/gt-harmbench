"""RQ3 judge rerun: Claude Sonnet 5.5 on all release reasoning traces via the
Anthropic Message Batches API (50% price), with the frozen judge configuration
(human_validation/RUBRIC.md, two-step scaffold, effort medium) — the same prompt
text that scored kappa 0.74 against Isabel's labels. The rubric is a cached
prefix (identical in every request).

  test:     PYTHONPATH=. uv run python -m analysis.reviewer_response.judge_batch test
  submit:   PYTHONPATH=. uv run python -m analysis.reviewer_response.judge_batch submit
  status:   PYTHONPATH=. uv run python -m analysis.reviewer_response.judge_batch status
  collect:  PYTHONPATH=. uv run python -m analysis.reviewer_response.judge_batch collect

Keys: ANTHROPIC_API_KEY from .env (key "a") and .env2 (key "b"); traces are split
40/60 across them in proportion to their credit.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import dotenv_values

from analysis.reviewer_response import judge_pilot as jp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/reviewer_response/data/judge_full"
STATE = OUT / "batches.json"
MODEL, EFFORT, MAX_TOKENS, CHUNK = "claude-sonnet-5-5", "medium", 8000, 1500
SPLIT_A = 0.40


def client(k):
    env = dotenv_values(ROOT / (".env" if k == "a" else ".env2"))
    return anthropic.Anthropic(api_key=env["ANTHROPIC_API_KEY"], max_retries=5)


def params(rubric, t):
    full = jp.build_prompt(rubric, t, "twostep")
    head = f"{rubric}\n\n---\n\n"
    assert full.startswith(head)
    return MessageCreateParamsNonStreaming(
        model=MODEL, max_tokens=MAX_TOKENS, output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": head, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": full[len(head):]}]}])


def traces():
    return jp.release_only(jp.load_traces())


def cmd_test():
    rubric = (jp.HV / "RUBRIC.md").read_text(); T = traces()[:2]; c = client("a")
    for t in T:
        r = c.messages.create(**params(rubric, t))
        text = next((b.text for b in r.content if b.type == "text"), "")
        labels, _, _ = jp.parse_labels(text, t["reasoning"])
        u = r.usage
        print(r.stop_reason, "| in", u.input_tokens, "cache_write", u.cache_creation_input_tokens,
              "cache_read", u.cache_read_input_tokens, "out", u.output_tokens, "| labels", sum(labels.values()))


def cmd_submit():
    OUT.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        sys.exit("batches.json exists — already submitted")
    rubric = (jp.HV / "RUBRIC.md").read_text(); T = traces()
    idmap = {f"t{i}": [t["id"], t["model"]] for i, t in enumerate(T)}
    cut = int(len(T) * SPLIT_A); state = {"idmap": idmap, "batches": []}
    for k, lo, hi in [("a", 0, cut), ("b", cut, len(T))]:
        c = client(k)
        for s in range(lo, hi, CHUNK):
            reqs = [Request(custom_id=f"t{i}", params=params(rubric, T[i])) for i in range(s, min(s + CHUNK, hi))]
            b = c.messages.batches.create(requests=reqs)
            state["batches"].append({"key": k, "id": b.id, "n": len(reqs), "range": [s, min(s + CHUNK, hi)]})
            print(f"key {k}: batch {b.id} with {len(reqs)} requests")
            STATE.write_text(json.dumps(state))


def cmd_status():
    st = json.loads(STATE.read_text())
    for b in st["batches"]:
        r = client(b["key"]).messages.batches.retrieve(b["id"]); rc = r.request_counts
        print(f"{b['key']} {b['id']} {r.processing_status:11s} done {rc.succeeded}/{b['n']} err {rc.errored} exp {rc.expired}")


def cmd_collect():
    st = json.loads(STATE.read_text()); idmap = st["idmap"]
    byid = {(t["id"], t["model"]): t for t in traces()}
    out = OUT / f"sonnet55_release__{len(idmap)}.jsonl"; n = bad = 0
    usage = {"in": 0, "cw": 0, "cr": 0, "out": 0}
    with open(out, "w") as f:
        for b in st["batches"]:
            c = client(b["key"])
            if c.messages.batches.retrieve(b["id"]).processing_status != "ended":
                print("not finished:", b["id"]); continue
            for res in c.messages.batches.results(b["id"]):
                tid, model = idmap[res.custom_id]; t = byid[(tid, model)]
                if res.result.type != "succeeded":
                    bad += 1; f.write(json.dumps({"id": tid, "model_judged": model, "error": res.result.type}) + "\n"); continue
                m = res.result.message; u = m.usage
                usage["in"] += u.input_tokens; usage["cw"] += u.cache_creation_input_tokens or 0
                usage["cr"] += u.cache_read_input_tokens or 0; usage["out"] += u.output_tokens
                text = next((x.text for x in m.content if x.type == "text"), "")
                labels, quotes, verified = jp.parse_labels(text, t["reasoning"])
                f.write(json.dumps({"id": tid, "model_judged": model, "game": t["formal_game"],
                                    "util_optimal": int(t["utilitarian_score"] == 1.0), "labels": labels,
                                    "quotes": quotes, "quote_verified": verified, "stop_reason": m.stop_reason,
                                    "raw": text}) + "\n"); n += 1
    cost = 0.5 * (usage["in"] * 2 + usage["cw"] * 2.5 + usage["cr"] * 0.2 + usage["out"] * 10) / 1e6
    print(f"{n} ok, {bad} failed -> {out}\nusage {usage}  est. cost ${cost:.2f}")


if __name__ == "__main__":
    {"test": cmd_test, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[sys.argv[1]]()
