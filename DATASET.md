# GT-HarmBench — Datasheet

This document describes `data/gt-harmbench-cleaned.csv`, the GT-HarmBench release dataset.

## At a glance

| | |
|---|---|
| Scenarios | 1,535 |
| Games | 6 canonical 2×2 symmetric games |
| Source taxonomy | [MIT AI Risk Repository](https://airisk.mit.edu/) |
| Distinct MIT risk entries covered | 578 |
| Structural validity | 1,530 / 1,535 (99.67%) |
| MIT provenance | 1,535 / 1,535 (100%) |
| Language | English |
| Licence | CC BY 4.0 (data) |

## Game distribution

| Game | n | % |
|---|---:|---:|
| Prisoner's Dilemma | 490 | 31.9% |
| Chicken | 379 | 24.7% |
| Stag Hunt | 317 | 20.7% |
| Coordination | 180 | 11.7% |
| Bach or Stravinski | 141 | 9.2% |
| No Conflict | 28 | 1.8% |

Matching Pennies is classified during generation but **excluded** from the release: it is zero-sum, so it has no socially optimal outcome to score against, which makes it meaningless for this benchmark's central question.

**A note on Stag Hunt ordering.** GT-HarmBench uses the payoff ordering `a > d > b > c`, not the `a > b > d > c` ordering given on Wikipedia. Both are valid Stag Hunts (the defining property — two Pareto-ranked pure Nash equilibria, with cooperation risky — holds either way). This is intentional and consistent across the dataset, including the abstract RQ2 variant.

## Domain coverage

Every scenario is anchored to a MIT AI Risk Repository entry, so this is the distribution of the entire dataset:

| Domain | n | % |
|---|---:|---:|
| 7. AI System Safety, Failures, & Limitations | 403 | 26.3% |
| 6. Socioeconomic and Environmental | 279 | 18.2% |
| 1. Discrimination & Toxicity | 257 | 16.7% |
| 4. Malicious Actors & Misuse | 242 | 15.8% |
| 5. Human-Computer Interaction | 143 | 9.3% |
| 2. Privacy & Security | 134 | 8.7% |
| 3. Misinformation | 77 | 5.0% |

## Schema

30 columns, in four groups.

**Provenance** — links back to the source taxonomy.

| Column | Description |
|---|---|
| `id` | Row identifier, unique within this file |
| `Ev_ID` | MIT AI Risk Repository entry ID |
| `Domain` | MIT top-level domain (7 categories) |
| `Risk category`, `Risk subcategory` | MIT taxonomy labels |
| `Description` | The source risk description |
| `justification` | Why this risk was judged game-theoretic |

**Scenario** — the narrative and its game.

| Column | Description |
|---|---|
| `formal_game` | One of the six game types |
| `story_row`, `story_col` | First-person narratives (~300 words) for each player |
| `actions_row`, `actions_column` | The two available actions per player |
| `1_1_payoff` … `2_2_payoff` | Payoff cells as `[row_payoff, col_payoff]`, each in [−10, 10] |
| `risk_level` | Severity, 1–10 (observed range 3–10, mean 6.73) |

**Generation QA** — retained for transparency about how each row was vetted.

| Column | Description |
|---|---|
| `quality_score`, `quality_justification` | Narrative quality, 0–10 (≥ 8 required to pass) |
| `equilibria_score`, `equilibria_justification` | Game-structure correctness, 0–10 (≥ 8 required) |
| `issues`, `overall_comment` | Free-text notes from the automated evaluation pass |

**Targets** — precomputed ground truth for scoring.

| Column | Description |
|---|---|
| `target_nash_equilibria` | Nash equilibrium action(s) |
| `target_utility_maximizing` | Action maximizing the payoff sum |
| `target_rawlsian` | Action maximizing the minimum payoff |
| `target_nash_social_welfare` | Action maximizing the payoff product |
| `nash_social_welfare`, `max_utilitarian`, `max_rawlsian` | The corresponding optimal values |

## How it was built

Three stages, all reproducible from `scripts/generation/`.

**Stage 1 — risk-to-game mapping.** The MIT repository is filtered to `Category level ∈ {Risk Category, Risk Sub-Category}` (1,612 rows). GPT-5.1 judges whether each risk is fundamentally game-theoretic and which of seven formal games fit it. 822 of 1,612 entries qualified, yielding 2,491 (risk, game) pairs — a mean of 3.04 games per strategic risk.

**Stage 2 — scenario generation.** For each pair, GPT-5.1 (`reasoning_effort="high"`) writes the paired narratives, action labels, and payoff matrix. A separate GPT-5.1 pass (`reasoning_effort="medium"`) scores quality and game-structure correctness; both must reach 8/10. Pass rate 80.7%, giving 2,009 scenarios. The full prompt is in `scripts/generation/generation_prompt.md`.

**Stage 3 — targets and cleaning.** `scripts/add_targets.py` computes the Nash, utilitarian, Rawlsian, and Nash-social-welfare targets. The cleaning step then removes 474 scenarios, producing the 1,535-row release (see below).

## What cleaning removed, and why

The unfiltered dataset has 2,009 scenarios (excluding Matching Pennies). 474 of them were dropped, leaving 1,535.

Those 474 came from a legacy pre-MIT classification run that was appended to `mit_game_fit.csv` during a pipeline consolidation. They carry valid model outputs and are game-theoretically sound (99.8% verifier pass rate), but they have empty `Ev_ID`/`Risk category`/`Description` — they are general strategic dilemmas with no anchor in a specific documented AI risk.

Because grounding in a real risk taxonomy is the point of this benchmark, they were removed rather than kept. The practical consequence is the property stated up front: **every scenario in the release traces to a MIT AI Risk Repository entry.**

An unfiltered 2,265-row file exists in the project history but is deliberately **not** part of this release, and no code here reads it.

## Validation

**Structural verifier.** `is_valid_game()` in `scripts/generation/fix_contextualizations.py` checks each payoff matrix against the canonical ordinal conditions for its game type, handling both action-labelling orientations. It is an independent audit, not a generation filter — nothing was silently repaired to satisfy it.

On the release: **1,530 / 1,535 pass (99.67%)**. The 5 failures are 4 Bach or Stravinski and 1 Chicken, all genuine ordinal outliers. They are retained so the dataset reflects what the pipeline actually produced. To exclude them, filter with `is_valid_game()` directly.

**Human inter-rater agreement.** Two annotators independently labelled the game type of 30 scenarios stratified across the six games (5 each, seed 42), choosing from 7 options including Matching Pennies:

- Cohen's κ = **0.84**, raw agreement 86.7% (26/30)

A separate binary good/bad quality rating on 100 scenarios (50 drawn from the 'good' pool, 50 from the 'bad' pool; seed 42) reached **Cohen's κ = 0.42**, raw agreement 73/100 — moderate, and noticeably weaker than the game-type agreement. Treat the `quality_score` column as a coarse signal, not a precise one.

Annotator ratings are in `artifacts/validation/`; run `python scripts/validation/compute.py` to reproduce both figures from those files.

## Known limitations

- **Payoffs are model-authored.** The matrices were written by GPT-5.1 to satisfy an ordinal structure, not elicited from human stakeholders. They encode the correct game *shape*; the specific magnitudes carry no calibrated real-world meaning.
- **Uneven game coverage.** No Conflict has only 28 scenarios — too few for confident per-game claims. Prisoner's Dilemma is over-represented at 31.9%, reflecting how often AI risks map onto free-rider structures rather than any sampling choice.
- **One risk yields several scenarios.** 1,535 scenarios come from 578 distinct MIT entries (mean 2.7 each). Scenarios sharing an `Ev_ID` are not statistically independent; group by `Ev_ID` when computing confidence intervals.
- **Generator and evaluator overlap.** Scenarios were generated *and* quality-scored by GPT-5.1. Evaluating GPT-5.1 on this benchmark carries a plausible family advantage. The human agreement study above is the main external check on that.
- **English only, single cultural frame.** Narratives are English and largely reflect Western institutional settings.
- **Two-player, single-shot, symmetric only.** No repeated play, no reputation, no coalitions, more than two actors, or asymmetric information.

## Provenance and licensing

Derived from the [MIT AI Risk Repository](https://airisk.mit.edu/), distributed under CC BY 4.0. GT-HarmBench data is released under **CC BY 4.0** in turn; the code in this repository is released under the **MIT Licence** (see `LICENSE`).

If you redistribute or build on this dataset, please retain attribution to both GT-HarmBench and the MIT AI Risk Repository.
