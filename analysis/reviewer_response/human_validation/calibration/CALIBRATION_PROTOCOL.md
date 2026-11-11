# Round-2 calibration protocol

Two annotators (Isabel, Amelia), one rubric (`../RUBRIC.md`, frozen 2026-10-22),
twelve calibration traces, then the pre-registered core 60. Both do the full
validation; Isabel remotely. Pepijn runs the calibration meeting and does not
annotate in round 2.

## Why a calibration round

Round 1 failed on definitional scope, not on care (pooled κ 0.19; UTILITARIAN
had zero overlap across 22 ticks). The literature is unanimous that a short
pilot, a discussion of every disagreement, and a frozen rubric before the scored
run is what produces agreement (Habernal & Gurevych 2017; Daffara et al. 2025;
Bayerl & Paul 2011: trained 81% vs untrained 70%). See `../RUBRIC_LITERATURE_NOTES.md`.

## The twelve traces

`calibration_set.csv`, fixed random order C01–C12. Drawn from the 104-trace
judge pilot, so **none is in the core 60 or the enrichment 20**. Every trace was
labelled identically by three LLM judges (Sonnet 5.5, gpt-5.1, Grok 4.7) under
this rubric, so each has a rubric-consistent answer to compare against. The set
is built to teach the rubric, not to probe its grey areas:

| Purpose | Traces |
|---|---|
| clear positives: all judges tick RAWLSIAN / AI_SAFETY / CATASTROPHE / UTILITARIAN | C01, C02, C04, C11 |
| pure game-theoretic reasoning; nothing else counts | C05, C12 |
| trap: "catastrophic / disaster" only inside the payoff description | C08 |
| trap: regulators, press, reputation hitting the player's own firm | C03 |
| trap: "ethical / responsible" with no specific consideration | C07 |
| rule: dominance stated and then overridden (NASH yes, PAYOFF_MAX no) | C09 |
| one long (~10k chars) and one short trace | C06, C10 |

Total reading: about 39k characters, roughly 8k words.

Judge labels live in `analysis/reviewer_response/data/human_validation_keys/_calibration_key.csv`,
outside the annotators' folder. Pepijn brings it to the meeting.

## Procedure

1. **Read.** Each annotator reads `RUBRIC.md` once (about 20 minutes). The
   quick-reference table on the first page is the working summary.
2. **Label independently.** `uv run streamlit run scripts/validation/reasoning_app.py`,
   choose *calibration*. For every tick: paste the sentence, and say whether it is
   restatement / mention / weighed. Only *weighed* counts. About 45 minutes to an hour.
   No discussion with the other annotator during this step.
3. **Compare.** Pepijn runs
   `uv run python analysis/reviewer_response/calibration_score.py` → per-category
   agreement (raw, κ from marginals, AC1) between the two annotators and each
   annotator vs the three-judge majority, plus a disagreement sheet with
   everyone's quotes side by side.
4. **Discuss every disagreement** (about 60 minutes, all three present). For each:
   - If the rubric already settles it, the rubric wins; note which sentence.
   - If the rubric is silent or ambiguous, write the one-sentence rule that
     settles it and the trace that triggered it.
   - If both readings are defensible and the rubric cannot be sharpened, record
     it in `HARD_CASES.md` as a known ambiguity.
5. **Amend and freeze.** Rule additions go into `RUBRIC.md` and are logged in
   `../PREREGISTRATION.md` under *Deviations* with date, trigger and text. Then
   the rubric is frozen for the scored run. Annotators may keep `HARD_CASES.md`
   open while labelling the core 60.
6. **Scored run.** Both label the core 60 under the frozen rubric (choose *core*
   in the app), independently, no discussion until both are done. Output goes to
   `annotation_<name>_r2.csv`; round-1 files are untouched.

## What we are and are not doing

- We **are** aligning the two annotators on definitions, which is what the
  reviewer asked for ("calibrate the classifier").
- We are **not** tuning the annotators to the judge. The judge labels are shown
  only in step 4, as a third reading to argue against, and only after both
  humans have committed. Agreement with the judge on the core 60 is the thing
  being measured; it must not be manufactured here.
- The twelve calibration traces are excluded from every reported number.

## Scheduling

Both annotators do the full validation; Isabel works remotely.

| By | Step |
|---|---|
| **Mon 26 Oct** | both: rubric read + 12 calibration traces labelled independently |
| Mon 26 / Tue 27 Oct | one-hour calibration meeting (video), rubric amended and frozen |
| **Fri 30 Oct** | both: core 60 labelled independently under the frozen rubric |
| Fri 30 / Sat 31 Oct | scored; round-2 results written; feeds the human-validation appendix |

If the calibration labels are late, the meeting slips but the core-60 date holds.
