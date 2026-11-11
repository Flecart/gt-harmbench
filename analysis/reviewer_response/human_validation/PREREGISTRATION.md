# Pre-registration — human validation of the RQ3 reasoning-trace classifier

Fixed **2026-07-31, before any human labels existed.** Everything below was
decided against the LLM judge's own labels only; no human annotation had been
collected, and none of these choices can have been tuned to a human result.

## What is being validated

RQ3 classifies 16,039 reasoning traces into 8 categories with a gpt-4o-mini
judge, and reports that welfare- and safety-oriented reasoning is more prevalent
when a decision is socially optimal while payoff-maximising reasoning is more
prevalent when it is not. Reviewer 5XrX noted the classifier was never
calibrated. This is the calibration.

## The promise being discharged

From the posted reply (`POST_5XrX_comment1.md`):

> Following the protocol we used for game-type classification, a stratified
> sample of traces, balanced across optimal and suboptimal outcomes, will be
> labeled independently by two annotators.
>
> Alongside inter-annotator agreement, we will report the association between
> each reasoning category and outcome optimality as an odds ratio, and confirm
> that the human-labeled data reproduce the direction we report.

Note what "the direction we report" refers to: the sentence preceding it defines
it as the two-part coarse claim (welfare/safety up, payoff-max down), not as a
per-category result. Per-category odds ratios are *reported*; the coarse
direction is what is *confirmed*.

## Sample

**Core, n=60** — outcome-balanced (30 optimal / 30 suboptimal), stratified
across the 4 models, game type left proportional to the corpus. Drawn seed 42,
committed 2026-07-28 (`0cdbacd`), before any of the analysis in this document.

Game type is deliberately **not** stratified. Game is a strong effect modifier:
across all 16,039 traces the PAYOFF_MAX odds ratio runs from 0.15 (Prisoner's
Dilemma) to 1.06 (Coordination) and 1.05 (No Conflict), and the welfare-category
gap reverses sign in Coordination — the effect lives in the social dilemmas,
because elsewhere the socially optimal action is also the individually obvious
one. The paper's odds ratio is pooled over the natural game mix, so
equal-weighting games estimates a different quantity than the claim under
validation (No Conflict is 1.9% of the corpus and would become 16.7% of the
sample). Balancing on the *outcome* is free — case-control sampling leaves an
odds ratio invariant — but stratifying on an effect modifier is not. A
game-stratified variant was drawn on 2026-07-31 and rejected on these grounds;
it is not used anywhere.

**Enrichment, n=20** — supplementary and optional for the annotator. 5
judge-positive traces each for UTILITARIAN, RAWLSIAN, AI_SAFETY, PRECAUTIONARY;
these low-prevalence categories yield ~10 positives in the core, too few to
estimate the judge's false-positive rate. Used for agreement only, **never** for
an odds ratio, because it is deliberately not outcome-representative.

**Blinding.** Annotators never see the model's choice or whether it was optimal.
That information exists only in `_llm_key.csv`. This is what prevents an
annotator who knows the RQ3 hypothesis from reproducing it by construction.

## Analyses, in order of evidential weight

### 1. Differential error (primary)

Human-vs-judge per-label disagreement rate computed separately within optimal
and within suboptimal traces, difference reported with a 2,000-sample bootstrap
CI.

This is the primary analysis because it is the one that can actually defend or
sink RQ3. Uniform judge error attenuates an association toward the null — it
cannot create one. Only outcome-correlated judge error can manufacture the
reported direction. A CI covering zero rules that out, and does so even if κ is
mediocre and every odds-ratio CI is wide.

### 2. Reliability

Per-category Cohen's κ (expected agreement from the raters' marginals) with
bootstrap CI, reported **alongside** prevalence, raw agreement, Gwet's AC1 and
PABAK. All four are reported for every category, always together.

Committed in advance because κ is known to be degenerate at extreme prevalence,
and NASH_DOMINANT sits at ~95%. On synthetic annotators with 12% pure random
noise it already yields κ = 0.18 against AC1 = 0.82 and 85% raw agreement.
Reporting κ alone there would misrepresent the task's reliability. We will not
lead on a single macro-κ.

### 3. Length split

Agreement recomputed for traces above and below the median reasoning length
(traces range 466 to ~34,700 characters). The judge read every trace in full; a
fatigued human may not have. If agreement is stable across the split, skimming
is ruled out; if it degrades on long traces, the human is the weaker rater there
and we will say so rather than attribute the gap to the judge.

### 4. Direction

- **Primary estimator**: number of welfare/safety categories per trace, optimal
  vs suboptimal, Mann-Whitney. Chosen over the binarised version because
  ~70–80% of traces carry at least one such category, so "any" hits a ceiling
  and discards the effect. On judge labels the count gives p = 0.043 where the
  binary odds ratio spans 1.
- **Secondary**: odds ratio for "any welfare/safety category present".
- **Reported for completeness**: per-category odds ratios with Woolf CIs, as
  promised. These are underpowered at n = 60 for the low-prevalence categories
  and the CIs will show it.
- **Consensus rule**: a category is present when **both** annotators tick it.
  Union is reported as a sensitivity analysis only, because it inflates
  prevalence.
- **"Welfare and safety oriented"** = UTILITARIAN, RAWLSIAN,
  CATASTROPHE_PREVENTION, AI_SAFETY, PRECAUTIONARY. PRECAUTIONARY is included
  despite being non-significant in the paper's judge-1 column, because it is
  substantively part of the family and excluding it after the fact would be a
  researcher degree of freedom.
- **"Reproduces the direction"** = the human-labelled primary estimator points
  the same way as the judge's on the same rows, and per-category odds-ratio
  *signs* agree between human and judge labels. The comparison is human-vs-judge
  **on the same 60 traces**, not human-vs-the-paper's-16,039, because n = 60
  cannot resolve the full-corpus effect sizes and any discrepancy would reflect
  sampling, not labelling.

## What would count as a negative result

Stated in advance so this cannot be a formality:

- Differential-error CI **excluding** zero → judge error is outcome-correlated;
  the RQ3 direction is potentially an artifact and we report that, per-category.
- Annotator-vs-judge agreement no better than annotator-vs-annotator → the judge
  is not tracking a construct humans share, and the taxonomy is the problem.
- Primary estimator pointing the **opposite** way under human labels → the
  direction does not survive human labelling and we say so plainly.
- κ and AC1 both low across categories → the categories are not reliably
  distinguishable and RQ3's granularity is overstated.

Any of these gets reported. We are not conditioning what we post on which way it
comes out.

## Deviations

Any departure from this document gets recorded here with its date and reason,
including the trigger. The game-stratified draw described above is the first
such entry: proposed and rejected on 2026-07-31, before annotation began.
