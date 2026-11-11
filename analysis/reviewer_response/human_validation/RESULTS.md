# Results — human validation of the RQ3 reasoning-trace classifier

Round 1, scored 2026-08-03 against `PREREGISTRATION.md` (fixed 2026-07-31).
Reproduce with:

```bash
uv run --with scipy python analysis/reviewer_response/human_validation_analysis.py
```

Both annotators completed the **core 60**. Neither did the optional
enrichment 20, so the low-prevalence categories stay underpowered as
anticipated.

## Headline

The pre-registered **primary** analysis passes. The pre-registered
**reliability** analysis fails, and fails hard enough that the primary result
cannot carry as much weight as intended.

| Pre-registered negative result | Triggered? |
|---|---|
| Differential-error CI excludes zero | **No** — covers zero for both annotators |
| Annotator-vs-judge no better than annotator-vs-annotator | **No** — judge agreement is higher for both |
| Primary estimator points the opposite way | **No** — every label set points the same way |
| κ and AC1 both low across categories | **Yes** |

## 1. Differential error (primary)

Human-vs-judge per-label disagreement, computed separately within optimal and
within suboptimal traces:

| Annotator | optimal | suboptimal | difference | 95% CI |
|---|---:|---:|---:|---|
| Pepijn | 0.217 | 0.167 | +0.050 | [−0.033, +0.129] |
| Angelo | 0.367 | 0.317 | +0.050 | [−0.029, +0.129] |

Both cover zero. (The identical point estimates are a coincidence: both
annotators happen to disagree with the judge on exactly 12 more labels out of
240 in optimal traces than in suboptimal ones.)

This is the analysis pre-registered as the one that can defend or sink RQ3, and
it defends it: judge error is not detectably outcome-correlated, so it cannot
have manufactured the reported direction. Uniform error only attenuates toward
the null. The caveat is power, and it is a real one — n=60 with an unreliable
reference standard cannot exclude a modest differential.

## 2. Reliability — the negative result

| Category | prev. P | prev. A | raw | κ | κ 95% CI | AC1 | PABAK |
|---|---:|---:|---:|---:|---|---:|---:|
| NASH_DOMINANT | 0.98 | 0.68 | 0.67 | −0.03 | [−0.10, 0.00] | 0.54 | 0.33 |
| PAYOFF_MAX | 0.90 | 0.43 | 0.50 | 0.10 | [−0.03, 0.24] | 0.10 | 0.00 |
| UTILITARIAN | 0.25 | 0.12 | 0.63 | −0.19 | [−0.29, −0.08] | 0.48 | 0.27 |
| RAWLSIAN | 0.20 | 0.07 | 0.77 | 0.03 | [−0.14, 0.29] | 0.70 | 0.53 |
| CATASTROPHE_PREVENTION | 0.35 | 0.18 | 0.63 | 0.09 | [−0.15, 0.33] | 0.40 | 0.27 |
| PRECAUTIONARY | 0.27 | 0.28 | 0.58 | −0.04 | [−0.28, 0.22] | 0.31 | 0.17 |
| AI_SAFETY | 0.13 | 0.17 | 0.80 | 0.22 | [−0.10, 0.51] | 0.73 | 0.60 |
| DOMAIN_OTHER | 0.08 | 0.67 | 0.42 | 0.09 | [0.02, 0.18] | **−0.10** | −0.17 |
| **pooled** | 0.40 | 0.33 | 0.62 | **0.19** | | 0.30 | 0.25 |

Reporting all four statistics together was pre-registered precisely so that a
prevalence artifact could not be mistaken for a reliability failure. It is not
a prevalence artifact: κ and AC1 are low *together*, which is the pre-registered
signature of genuinely unreliable categories.

**The decisive test.** Shuffling one annotator's rows destroys every
trace-level correspondence while preserving exactly how often they tick each
label. If observed agreement sits inside that null, the two share a labelling
frequency and nothing else.

| Pair | observed | null mean | null 95% | p |
|---|---:|---:|---|---:|
| Pepijn vs Angelo | 0.625 | 0.610 | [0.583, 0.633] | **0.145** |
| Pepijn vs judge | 0.808 | 0.702 | [0.675, 0.733] | <0.0001 |
| Angelo vs judge | 0.658 | 0.603 | [0.575, 0.633] | 0.0002 |

**Inter-annotator agreement is not distinguishable from chance given the
marginals.** No individual category reaches p < 0.10 either. Both annotators
nonetheless agree with the judge above chance — the judge sits between the two
human readings rather than outside them, which is why negative result #2 did
not trigger.

Mechanical explanations were checked and ruled out: no column permutation (each
of Angelo's columns matches Pepijn's same-named column as well as any other,
after accounting for prevalence), and no row misalignment (pooled agreement
peaks at lag 0, but only barely — 0.625 against 0.59–0.62 at arbitrary lags,
which is itself the same finding).

## 3. What actually went wrong: definitional scope, not judgment

Disagreements are systematic and directional, not noisy. Positive-label overlap:

| Category | Pepijn | Angelo | both | Jaccard |
|---|---:|---:|---:|---:|
| NASH_DOMINANT | 59 | 41 | 40 | 0.67 |
| PAYOFF_MAX | 54 | 26 | 25 | 0.45 |
| UTILITARIAN | 15 | 7 | **0** | 0.00 |
| RAWLSIAN | 12 | 4 | 1 | 0.07 |
| CATASTROPHE_PREVENTION | 21 | 11 | 5 | 0.19 |
| PRECAUTIONARY | 16 | 17 | 4 | 0.14 |
| AI_SAFETY | 8 | 10 | 3 | 0.20 |
| DOMAIN_OTHER | 5 | 40 | 5 | 0.12 |

UTILITARIAN has **zero** overlap across 22 positive labels. DOMAIN_OTHER is a
near-inversion, with Pepijn's 5 a strict subset of Angelo's 40. Neither is
consistent with two people applying one definition imprecisely; both are
consistent with two people applying different definitions consistently.

Reading the traces identifies the split. Traces open by restating the
scenario's stakes, because that is what the prompt hands them:

> *"…could determine which of us survives… we each earn solid margins, pay down
> debt, and preserve jobs"* (#1)
>
> *"…will shape public perception, vaccination uptake, and revenue"* (#5)

Angelo counts restated stakes as DOMAIN_OTHER — the definition does say
"references domain stakes." Pepijn requires the reasoning to *invoke* those
stakes as a justification for the choice. Both are faithful readings of the
rubric as written. The same split explains UTILITARIAN: on #1 the trace ranks
outcomes for both players, which Angelo reads as aggregate-welfare framing and
Pepijn reads as ordinary payoff ranking.

This is the same failure mode as the scenario-echo problem in the
evaluation-awareness audit — prompt restatement being scored as the model's own
reasoning — and it was not anticipated in the rubric.

**On the gap between the two annotators.** Pepijn agrees with the judge
substantially better (raw 0.81, κ 0.60) than Angelo does (0.66, κ 0.26). We do
not have an explanation for this. Neither annotator wrote the classifier prompt
(Amelia did), so it is not authorship of the rubric. Both are co-authors who
know the RQ3 hypothesis, which is why the pre-registered design blinds them to
the model's choice and to optimality — that blinding, not naivety, is what
prevents either from reproducing the hypothesis by construction.

With two raters and n=60, one agreeing more closely with the classifier than the
other is a single observation and should not be read as evidence either way
about the classifier. It is reported because it is the largest asymmetry in the
data, not because it is interpretable.

## 3b. Each annotator against the judge

The judge is the object under validation, not a ground truth. "Over"/"under"
below mean only *ticked where the judge did not*, and vice versa.

| Category | judge | P | over | under | sens | spec | A | over | under | sens | spec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NASH_DOMINANT | 57 | 59 | 3 | 1 | 0.98 | 0.00 | 41 | 1 | 17 | 0.70 | 0.67 |
| PAYOFF_MAX | 45 | 54 | 11 | 2 | 0.96 | 0.27 | 26 | 3 | 22 | 0.51 | 0.80 |
| UTILITARIAN | 10 | 15 | 10 | 5 | 0.50 | 0.80 | 7 | 6 | 9 | 0.10 | 0.88 |
| RAWLSIAN | 10 | 12 | 7 | 5 | 0.50 | 0.86 | 4 | 4 | 10 | 0.00 | 0.92 |
| CATASTROPHE_PREVENTION | 35 | 21 | 1 | 15 | 0.57 | 0.96 | 11 | 2 | 26 | 0.26 | 0.92 |
| PRECAUTIONARY | 9 | 16 | 10 | 3 | 0.67 | 0.80 | 17 | 13 | 5 | 0.44 | 0.75 |
| AI_SAFETY | 15 | 8 | 2 | 9 | 0.40 | 0.96 | 10 | 4 | 9 | 0.40 | 0.91 |
| DOMAIN_OTHER | 9 | 5 | 2 | 6 | 0.33 | 0.96 | 40 | 32 | 1 | 0.89 | 0.37 |
| **POOLED** | **190** | **190** | **46** | **46** | 0.76 | 0.84 | **156** | **65** | **99** | 0.48 | 0.78 |

Pepijn ticks exactly as many labels as the judge (190 vs 190) with perfectly
balanced departures (46 over, 46 under) — no threshold offset. Angelo ticks
fewer (156) and departs asymmetrically. But neither is a uniform threshold
difference: both raters are *stricter* than the judge on some categories and
*looser* on others, which is again scope, not calibration.

Splitting all 480 label decisions three ways:

| | n | % |
|---|---:|---:|
| all three agree | 262 | 54.6 |
| judge sides with Pepijn against Angelo | 126 | 26.2 |
| judge sides with Angelo against Pepijn | 54 | 11.2 |
| both humans agree, judge differs | 38 | 7.9 |

The judge breaks Pepijn's way 2.3:1. **This is not in itself evidence that the
judge is right.** With one rater agreeing more than the other and no explanation
for why (see above), the 2.3:1 says the judge's reading is closer to one of two
available human readings — not that it is closer to a correct one. Had Angelo
been the closer rater the same statistic would have pointed the other way.

The last row is the only cell that isolates *judge* miscalibration from human
unreliability, since it is where both humans independently land on the same
side against the judge. It concentrates in the risk family:

| Category | judge alone ticks | both humans tick, judge does not |
|---|---:|---:|
| CATASTROPHE_PREVENTION | **11** | 0 |
| AI_SAFETY | 6 | 0 |
| RAWLSIAN | 5 | 1 |
| UTILITARIAN | 4 | 0 |
| PAYOFF_MAX | 1 | 3 |
| others | ≤2 | ≤2 |

The judge ticks CATASTROPHE_PREVENTION on 35 of 60 traces; Pepijn ticks 21 and
Angelo 11. It ticks PRECAUTIONARY on 9; the humans tick 16 and 17. Family
totals: judge 44, Pepijn 37, Angelo 28. On 7 of the 11 traces where the judge
alone ticks CATASTROPHE_PREVENTION, at least one human ticked PRECAUTIONARY
instead. That is suggestive — not conclusive at n=11 — that **the judge absorbs
precautionary reasoning into catastrophe-prevention**, a boundary both humans
draw further out. Both categories are inside the welfare/safety family, so this
particular error partly cancels in the primary estimator, which is consistent
with the differential-error result coming out clean.

## 3c. Is DOMAIN_OTHER driving the reliability failure?

It is the single worst category, and it is the one that matters least — it is
**not** in the welfare/safety family, so it enters no reported claim. But
dropping it does not rescue anything. Leave-one-out on pooled reliability:

| Dropped | raw | κ | AC1 | permutation p |
|---|---:|---:|---:|---:|
| — (all 8) | 0.62 | 0.19 | 0.30 | 0.145 |
| DOMAIN_OTHER | 0.65 | 0.27 | 0.36 | 0.302 |
| UTILITARIAN | 0.62 | 0.21 | 0.28 | 0.042 |
| NASH_DOMINANT | 0.62 | 0.08 | 0.35 | 0.122 |
| every other single category | 0.60–0.63 | 0.17–0.22 | 0.24–0.31 | 0.086–0.292 |

Dropping DOMAIN_OTHER lifts κ from 0.19 to 0.27 and *raises* the permutation p
from 0.145 to 0.302 — the remaining seven categories are further from
distinguishable-from-chance, not closer. No single category is driving the
result.

The decisive cut is by family:

| Subset | n | raw | κ | AC1 | permutation p |
|---|---:|---:|---:|---:|---:|
| welfare/safety family (the 5 that carry RQ3) | 300 | 0.68 | **0.03** | 0.53 | **0.46** |
| game-theoretic family (2) | 120 | 0.58 | 0.07 | 0.33 | 0.259 |
| the 7 excluding DOMAIN_OTHER | 420 | 0.65 | 0.27 | 0.36 | 0.302 |

**The five categories that actually carry RQ3's claim are the least reliable of
the set**: κ = 0.03, permutation p = 0.46. Their AC1 of 0.53 is prevalence-driven
— these categories are rare, so most cells are agreed absences. Excluding
DOMAIN_OTHER improves the headline number while leaving the substantive problem
untouched, and doing so after seeing the result would be a researcher degree of
freedom. The finding stands as stated in §2.

## 4. Length split

Median 3,256 chars (range 466–34,662).

| | A vs B raw | A vs B AC1 | Pepijn vs judge | Angelo vs judge |
|---|---:|---:|---:|---:|
| short (n=30) | 0.60 | 0.26 | 0.74 | 0.64 |
| long (n=30) | 0.65 | 0.35 | 0.88 | 0.68 |

Agreement is *higher* on long traces for every pair. Skimming under fatigue is
ruled out — longer traces state their reasoning more explicitly, which helps
both raters. This one is clean.

## 5. Direction

Primary estimator: count of welfare/safety categories per trace, optimal vs
suboptimal, Mann-Whitney.

| Labels | optimal | suboptimal | diff | p |
|---|---:|---:|---:|---:|
| judge | 1.60 | 1.03 | +0.57 | **0.044** |
| consensus (both tick) | 0.30 | 0.13 | +0.17 | 0.113 |
| union (sensitivity) | 1.97 | 1.63 | +0.33 | 0.347 |
| Pepijn alone | 1.40 | 1.00 | +0.40 | 0.243 |
| Angelo alone | 0.87 | 0.77 | +0.10 | 0.461 |

Every label set points the same way as the judge. None of the human ones reaches
significance at n=60, which is expected and was pre-registered as expected. Per
the pre-registered definition, "reproduces the direction" requires the human
estimator to point the same way — it does.

Secondary, "any welfare/safety category present": judge OR 1.93 [0.62, 6.04];
consensus OR 2.97 [0.76, 11.61]. Per-category odds ratios agree in sign between
judge and consensus for 4 of 8 categories, but every consensus CI spans 1 — at
these prevalences n=60 cannot resolve them, as pre-registered.

## 5b. Sensitivity — how much do the conclusions depend on who labels?

Comparing human ORs on 60 rows against the paper's ORs on 16,039 conflates two
things: the labeller, and n=60. To separate them, draw 4,000 outcome-balanced
30/30 subsamples from the **judge's own** full-corpus labels and record the OR
each time. That band is what n=60 does on its own, with the labeller held fixed.

| Category | judge, full 16,039 | n=60 sampling band | Pepijn | Angelo | consensus | judge on our 60 |
|---|---:|---|---:|---:|---:|---:|
| NASH_DOMINANT | 0.75 | [0.13, 7.76] | 3.10 | 0.86 | 1.00 | 0.58 |
| PAYOFF_MAX | 0.37 | [0.10, 1.20] | **1.94\*** | 0.59 | 0.67 | 0.19 |
| UTILITARIAN | 1.69 | [0.41, 7.76] | 1.67 | 1.33 | 1.00 | 1.56 |
| RAWLSIAN | 1.96 | [0.56, 7.43] | **0.45\*** | 1.00 | 3.10 | 1.00 |
| CATASTROPHE_PREVENTION | 1.24 | [0.45, 3.35] | 2.75 | 1.23 | 0.69 | 2.58 |
| PRECAUTIONARY | 1.07 | [0.29, 4.24] | 1.38 | 1.60 | 2.50 | 1.27 |
| AI_SAFETY | 1.84 | [0.49, 7.44] | 3.02 | 0.64 | **7.76\*** | 3.47 |
| DOMAIN_OTHER | 1.54 | [0.42, 5.84] | 0.69 | 1.33 | 0.69 | 2.08 |

\* outside the band. Angelo 0/8, judge-on-our-60 0/8, consensus 1/8, Pepijn 2/8
(≈0.4/8 expected by chance at 5%).

**The bands are enormous.** AI_SAFETY has a full-corpus OR of 1.84, but a 60-row
draw of the judge's *own* labels lands anywhere in [0.49, 7.44]. Per-category
odds ratios are not identifiable at n=60 for *any* labeller, the judge included.
The 4-of-8 sign concordance in §5 is therefore close to what sign concordance
would look like if the labels were identical and only the rows differed — it is
not evidence about labelling.

The coarse claims — the two actually promised — are what this can speak to:

| Label set | welfare/safety count diff | any-welfare OR | PAYOFF_MAX OR |
|---|---:|---:|---:|
| judge, full 16,039 | +0.33 | 1.95 | 0.37 |
| *n=60 sampling band* | *[−0.13, +0.80]* | *[0.60, 6.90]* | *[0.10, 1.20]* |
| judge on our 60 | +0.57 | 1.93 | 0.19 |
| Pepijn | +0.40 | 1.70 | **1.94** |
| Angelo | +0.10 | 1.95 | 0.59 |
| consensus | +0.17 | 2.97 | 0.67 |

Every label set reproduces both coarse claims inside the sampling band, with one
exception: **Pepijn's PAYOFF_MAX OR inverts** (1.94 against a full-corpus 0.37)
and falls outside the band.

That exception is a ceiling effect, not a contradiction. PAYOFF_MAX splits:

| | optimal | suboptimal | prevalence |
|---|---:|---:|---:|
| judge (full corpus) | | | 0.76 |
| judge (our 60) | 18/30 | 27/30 | 0.75 |
| Angelo | 11/30 | 15/30 | 0.43 |
| Pepijn | 28/30 | 26/30 | **0.90** |

At 90% prevalence there is almost no variance left to correlate with outcome, so
the estimate is uninformative rather than opposed. Angelo lands at 0.59 — same
direction as the judge, comfortably inside the band — and his any-welfare OR of
1.95 is identical to the full-corpus value.

**Bottom line.** RQ3's reported direction is *not* sensitive to who labels. This
substantially qualifies §2: the categories are unreliable at the boundary, yet
the coarse claim survives every way of drawing that boundary — unreliable labels
plus a surviving direction is evidence the direction does not depend on the
taxonomy's fine structure. What the evidence does not support is any
*per-category* claim at this sample size, from the humans or from the judge.

## 5c. The disagreement is systematic in scope, orthogonal to outcome

§3 established that the annotators diverge systematically, not randomly — two
readings of the rubric, applied consistently. That sounds worse for RQ3 than
random noise, and for a per-category claim it is. For the reported claim it is
not, provided the divergence is unrelated to the thing being measured.

Tested directly — is disagreement itself associated with outcome optimality?

| Disagreement measure | optimal | subopt. | difference | 95% CI | p |
|---|---:|---:|---:|---|---:|
| Pepijn vs Angelo, all 8 | 0.400 | 0.350 | +0.050 | [−0.029, +0.133] | 0.30 |
| Pepijn vs Angelo, welfare/safety 5 | 0.333 | 0.300 | +0.033 | [−0.080, +0.147] | 0.64 |
| Pepijn vs judge, all 8 | 0.217 | 0.167 | +0.050 | [−0.033, +0.133] | 0.30 |
| Pepijn vs judge, welfare/safety 5 | 0.227 | 0.220 | +0.007 | [−0.093, +0.107] | 0.91 |
| Angelo vs judge, all 8 | 0.367 | 0.317 | +0.050 | [−0.029, +0.129] | 0.28 |
| Angelo vs judge, welfare/safety 5 | 0.333 | 0.253 | +0.080 | [−0.020, +0.180] | 0.14 |
| the DOMAIN_OTHER scope split | 0.633 | 0.533 | +0.100 | [−0.167, +0.333] | 0.60 |
| the UTILITARIAN scope split | 0.433 | 0.300 | +0.133 | [−0.100, +0.367] | 0.42 |

Eight tests, all null. Even the two scope splits that most damage κ — the ones
where the annotators are demonstrably reading different definitions — are
unrelated to whether the trace's decision was socially optimal.

This is the classical condition for **non-differential misclassification**, and
its consequence is attenuation: measured effects shrink toward the null but keep
their sign. That is visible in the per-category prevalence differences
(optimal − suboptimal), where the noisier rater consistently shows the judge's
sign at smaller magnitude:

| Category | judge | Pepijn | Angelo |
|---|---:|---:|---:|
| PAYOFF_MAX | −0.30 | +0.07 | −0.13 |
| CATASTROPHE_PREVENTION | +0.23 | +0.23 | +0.03 |
| AI_SAFETY | +0.23 | +0.13 | −0.07 |
| UTILITARIAN | +0.07 | +0.10 | +0.03 |
| PRECAUTIONARY | +0.03 | +0.07 | +0.10 |
| NASH_DOMINANT | −0.03 | +0.03 | −0.03 |

So the accurate description is **systematic in what counts as a category, random
with respect to what we measure** — not "mainly noise," which understates how
structured the disagreement is, and not "biased," which is what would actually
threaten RQ3.

Caveat on power, stated once and meant: eight null tests at n=60 are eight
underpowered tests. They exclude large outcome-correlated error, not modest
amounts. The argument is that no mechanism for manufacturing the effect has been
found where one would show up, not that none exists.

## 6. Assessment and recommendation

The honest summary is: **the judge's error is not outcome-correlated, the
direction survives every labelling, and the eight-category taxonomy is not
reliably distinguishable by two independent readers.** RQ3's coarse claim is
supported; its per-category granularity is not, on this evidence — and §5b shows
per-category claims are not identifiable at n=60 for the judge either, so that
is a limit of the validation design as much as of the taxonomy.

The three results are less in tension than they look, but the reconciliation has
to be stated carefully.

**Defensible:**

- The coarse direction holds under all four labellings for welfare/safety
  (4/4 positive) and three of four for payoff-max, with every value inside the
  band that n=60 sampling produces on the judge's own labels (§5b).
- Disagreement — annotator-vs-annotator and annotator-vs-judge, whole taxonomy
  and welfare subset — is unrelated to outcome across eight tests (§5c). This is
  the non-differential condition, under which judge error attenuates the
  measured effect and cannot manufacture it.
- The attenuation pattern is visible: the noisier rater shows the judge's sign
  at smaller magnitude on the categories with the largest effects.
- The full-corpus effect is estimated at n=16,039, where it is not marginal
  (p = 7e−115 on the primary estimator). An attenuated-but-real effect is
  consistent with everything above.

**Not defensible, and should not be claimed:**

- That the judge is *validated* per category. κ is low, the categories are not
  cleanly separable by two independent readers, and §5b shows per-category odds
  ratios are unidentifiable at n=60 for the judge as well as for the humans.
- That judge error *cannot* be outcome-correlated. The tests are null but
  underpowered; they exclude large differentials, not modest ones. A signed,
  paired version of the test puts the judge-attributable share of the n=60
  effect somewhere between 29% and 82% depending on the human reference — though
  that range is itself confounded, because unreliable human labels attenuate the
  human residual and inflate the judge's apparent share. The signed test cannot
  separate "judge inflates" from "human is noisy"; the unsigned pre-registered
  test can, which is why it was the pre-registered one.

The one-sentence version: **the classifier is imperfectly calibrated, its errors
show no relation to the outcome variable in any test we can run, and the reported
direction survives every labelling of the same traces — so the finding is
attenuated by judge noise rather than produced by it.**

Two options, both defensible:

1. **Report round 1 as it stands.** The preregistration commits to reporting
   whichever way it comes out, and this includes a real negative result about
   the taxonomy. Costs nothing further and is maximally honest.

2. **Run a calibration round and re-annotate.** The disagreement is
   concentrated in definitional scope on four categories (UTILITARIAN,
   DOMAIN_OTHER, CATASTROPHE_PREVENTION, PRECAUTIONARY), not in judgment. A
   short adjudication pass — decide whether restated stakes count, write two
   worked examples per category — plausibly fixes most of it, and the length
   split says the annotators were attentive. This is what
   `project_kappa_miscomputation.md` already flagged as the lesson from the
   game-type validation.

If (2), the deviation must be logged in `PREREGISTRATION.md` with date, reason
and trigger, and **both rounds reported**. Round 1 cannot be dropped: it was
pre-registered, it was run, and suppressing it after seeing the result is
exactly what the preregistration exists to prevent. Note also that a second
round is no longer blind to the hypothesis for either annotator.

Recommendation: **(2), with both rounds reported.** A reviewer who asked for
calibration is better served by "we calibrated, here is what it took, and here
is the uncalibrated baseline" than by a single κ = 0.19.

One caveat on what round 2 can buy. §5b shows n=60 cannot resolve per-category
odds ratios no matter who labels, so better agreement would not make those cells
interpretable — it would only strengthen the reliability claim and tighten the
differential-error CI. If the goal is per-category evidence, the binding
constraint is sample size, not agreement.

## Note on implementation

`analysis/reviewer_response/score_human_validation.py` (written independently in
a parallel session) implements the same four pre-registered analyses. Its output
matches `human_validation_analysis.py` on every shared quantity — the OR tables
are identical and the primary Mann-Whitney p agrees to rounding. The two scripts
should be consolidated; until then, agreement between them is a useful check.
The permutation null, leave-one-out, family cut and §5b sampling band are only in
`human_validation_analysis.py`.
