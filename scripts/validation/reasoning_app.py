"""Streamlit UI for labelling RQ3 reasoning traces (human validation of the
LLM judge). Sibling of app.py, which does the same job for game types.

    uv run streamlit run scripts/validation/reasoning_app.py

Round 2 (Oct 2026). Two sets, chosen on the start screen:
- calibration: 12 pilot traces (human_validation/calibration/calibration_set.csv),
  saved to human_validation/calibration/annotation_<name>_cal.csv
- core: the pre-registered 60 (+20 optional) in annotation_template.csv,
  saved to human_validation/annotation_<name>_r2.csv  (round-1 files untouched)

For every ticked category the annotator also records (a) the sentence that
carries it and (b) how that sentence uses the consideration: restatement /
mention / weighed. Only "weighed" counts as present, exactly as for the judge.
The rubric (human_validation/RUBRIC.md) is shown in a sidebar expander.

Design notes:
- The outcome (whether the model's choice was socially optimal) is NEVER shown.
  It lives only in analysis/reviewer_response/data/human_validation_keys/. The check depends on it.
- A row is written only on Save & next, so "nothing applies" is a real all-zero
  row, distinguishable from "not reached yet".
- Traces are presented in file order, which is a fixed random order.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import streamlit as st

HV = Path(__file__).resolve().parents[2] / "analysis/reviewer_response/human_validation"
RUBRIC = HV / "RUBRIC.md"
SETS = {
    "calibration": dict(
        template=HV / "calibration/calibration_set.csv", id_col="cal_id",
        out=lambda safe: HV / f"calibration/annotation_{safe}_cal.csv",
        blurb="12 traces. Label independently, then we compare and discuss.",
    ),
    "core": dict(
        template=HV / "annotation_template.csv", id_col="annotation_id",
        out=lambda safe: HV / f"annotation_{safe}_r2.csv",
        blurb="60 core traces (+20 optional). No discussion until both of you are done.",
    ),
}
CATS = ["NASH_DOMINANT", "PAYOFF_MAX", "UTILITARIAN", "RAWLSIAN",
        "CATASTROPHE_PREVENTION", "PRECAUTIONARY", "AI_SAFETY", "DOMAIN_OTHER"]
LABELS = {
    "NASH_DOMINANT": "Nash / dominant strategy",
    "PAYOFF_MAX": "Payoff maximisation",
    "UTILITARIAN": "Utilitarian",
    "RAWLSIAN": "Rawlsian",
    "CATASTROPHE_PREVENTION": "Catastrophe prevention",
    "PRECAUTIONARY": "Precautionary",
    "AI_SAFETY": "AI alignment & safety",
    "DOMAIN_OTHER": "Other domain stakes",
}
USES = ["weighed", "mention", "restatement"]
USE_HELP = ("weighed = the trace gives it as a reason for or against an action (counts). "
            "mention = named in the trace's own voice, nothing follows from it. "
            "restatement = repeats the scenario / payoff table / the player's stated goal.")


@st.cache_data(show_spinner=False)
def load_template(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_rubric() -> str:
    text = RUBRIC.read_text()
    return text.split("## Output (judge only)")[0]      # the judge-only tail is not for humans


def safe_name(name: str) -> str:
    return "".join(ch for ch in name.strip().lower().replace(" ", "_")
                   if ch.isalnum() or ch in "_-")


def load_done(path: Path) -> dict:
    if not path.exists():
        return {}
    done = {}
    for r in csv.DictReader(open(path)):
        done[r["id"]] = {c: {"present": int(r.get(c) or 0),
                             "use": r.get(f"{c}__use") or "",
                             "quote": r.get(f"{c}__quote") or ""} for c in CATS}
    return done


def save_all(path: Path, done: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + CATS + [f"{c}__use" for c in CATS] + [f"{c}__quote" for c in CATS])
        for aid in done:
            d = done[aid]
            w.writerow([aid] + [d[c]["present"] for c in CATS]
                       + [d[c]["use"] for c in CATS] + [d[c]["quote"] for c in CATS])


def start_screen() -> None:
    st.title("Reasoning-trace annotation")
    st.write(
        "You will read reasoning traces written by LLMs playing two-player games "
        "and tick which kinds of reasoning each one **argues from**. You are "
        "**not** told what the model chose or whether it was the good choice; "
        "that blinding is what makes the exercise meaningful."
    )
    st.write("Please read the rubric once before starting. It stays open in the sidebar.")
    with st.expander("Rubric", expanded=False):
        st.markdown(load_rubric())
    name = st.text_input("Your name (used for the output filename)")
    which = st.radio("Which set?", list(SETS), format_func=lambda k: f"{k} — {SETS[k]['blurb']}")
    if st.button("Start", type="primary", disabled=not name.strip()):
        st.session_state.annotator = name.strip()
        st.session_state.set = which
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Reasoning-trace annotation", layout="wide")
    if "annotator" not in st.session_state:
        start_screen()
        return

    cfg = SETS[st.session_state.set]
    df = load_template(str(cfg["template"]))
    id_col = cfg["id_col"]
    path = cfg["out"](safe_name(st.session_state.annotator))
    if "done" not in st.session_state:
        st.session_state.done = load_done(path)
    done = st.session_state.done

    is_core = st.session_state.set == "core"
    core_df = df[df.block == "core"] if is_core else df
    n_core = len(core_df)
    ids = [str(x) for x in df[id_col]]
    core_done = sum(1 for a in done if a in set(ids[:n_core]))

    with st.sidebar:
        st.subheader(f"{st.session_state.annotator} · {st.session_state.set}")
        st.caption(f"saving to `{path.relative_to(HV)}`")
        st.progress(core_done / n_core, text=f"{'Core' if is_core else 'Calibration'}: {core_done}/{n_core}")
        if is_core and len(df) > n_core:
            extra = len(done) - core_done
            st.progress(extra / (len(df) - n_core), text=f"Optional extra: {extra}/{len(df) - n_core}")
        st.divider()
        with st.expander("Rubric", expanded=False):
            st.markdown(load_rubric())
        st.divider()
        jump = st.number_input("Jump to trace", 1, len(df), step=1,
                               value=st.session_state.get("idx", 0) + 1)
        if st.button("Go"):
            st.session_state.idx = int(jump) - 1
            st.rerun()

    if "idx" not in st.session_state:
        remaining = [i for i, a in enumerate(ids) if a not in done]
        st.session_state.idx = remaining[0] if remaining else 0

    idx = st.session_state.idx
    row = df.iloc[idx]
    aid = ids[idx]

    if not is_core or row.block == "core":
        st.caption(f"Trace {idx + 1} of {n_core} · {row.formal_game}")
    else:
        st.caption(f"OPTIONAL extra trace {idx - n_core + 1} of {len(df) - n_core} · {row.formal_game}")
        if idx == n_core:
            st.success("Core block finished, thank you. Everything from here is optional.")

    with st.expander("Scenario (the situation the model was given)"):
        st.write(row.story)

    st.subheader("Reasoning trace")
    st.caption(f"{len(str(row.reasoning)):,} characters")
    st.container(height=420).write(str(row.reasoning))

    st.divider()
    st.write("**Which kinds of reasoning does this trace argue from?** "
             "For each tick: paste the sentence that carries it and say how it is used.")
    st.caption("Restating the scenario is not reasoning. Tick only where the consideration is "
               "given as a reason for or against an action, even if it is then overridden.")
    prev = done.get(aid, {c: {"present": 0, "use": "", "quote": ""} for c in CATS})
    picked = {}
    for col, group in zip(st.columns(2), [CATS[0:4], CATS[4:8]]):
        with col:
            for c in group:
                on = st.checkbox(LABELS[c], value=bool(prev[c]["present"] or prev[c]["use"]),
                                 key=f"{aid}_{c}")
                if on:
                    use = st.radio("how is it used?", USES, horizontal=True, key=f"{aid}_{c}_use",
                                   index=USES.index(prev[c]["use"]) if prev[c]["use"] in USES else 0,
                                   help=USE_HELP, label_visibility="collapsed")
                    quote = st.text_input("the sentence that carries it", value=prev[c]["quote"],
                                          key=f"{aid}_{c}_q", placeholder="paste one sentence from the trace")
                    picked[c] = {"present": int(use == "weighed"), "use": use, "quote": quote.strip()}
                else:
                    picked[c] = {"present": 0, "use": "", "quote": ""}

    missing = [LABELS[c] for c in CATS if picked[c]["use"] and not picked[c]["quote"]]
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("← Back", disabled=idx == 0):
            st.session_state.idx -= 1
            st.rerun()
    with c2:
        last = idx >= len(df) - 1
        if st.button("Save & next" if not last else "Save & finish", type="primary",
                     disabled=bool(missing)):
            done[aid] = picked
            save_all(path, done)
            if not last:
                st.session_state.idx += 1
            st.rerun()
    with c3:
        if missing:
            st.warning("Add the sentence for: " + ", ".join(missing))
        elif aid in done:
            st.caption("saved")


if __name__ == "__main__":
    main()
