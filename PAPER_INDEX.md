# Paper → code index

Where to look to check a number, table or figure in the GT-HarmBench camera-ready
(NeurIPS 2026). Rows follow the paper's order. Table/figure numbers are from the
camera-ready build; the LaTeX label is given too, since numbers shift between versions.

All commands run from the repo root:

```bash
uv sync
PYTHONPATH=. uv run python -m <module>
```

## What you need

| Input | Where | Needed for |
|---|---|---|
| Release dataset, 1,535 scenarios | `data/gt-harmbench-cleaned.csv` | everything |
| Abstract condition (1:1 with the release) | `data/gt-harmbench-abstract-full.csv` | RQ2 abstract |
| Gamified condition (payoffs shown) | `data/gt-harmbench-gamify.csv` | RQ2 framing |
| Mechanism variants | `data/mech_variants/` (20 CSVs) | RQ4 |
| Raw `.eval` logs | **not in git** — download from the [log archive on Google Drive](https://drive.google.com/drive/folders/1hyXR_iLqhZiUV8VdqUeI5eMZ6_nA_1iN) into `logs/` | every row marked *logs* |
| Log inventory | `analysis/reviewer_response/logs_manifest.csv` (every log: role, model, dataset, n, md5) | finding the right log |

The Drive folders are named differently from the local layout below (e.g.
`RQ1 - baseline/` on Drive). `logs_manifest.csv` maps each `drive_path` to the
`local_path` the scripts expect, and its `md5` column verifies the download.

Expected layout of `logs/` (the scripts read these paths):

```
logs/baseline/<model>.eval          RQ1, one pinned log per model (list in reproduce_tables.py: LOGS)
logs/RQ2 - gamified/                RQ2 framing (payoffs shown)
logs/RQ2-coordination/              RQ2 order effect (random option order)
logs/abstract/                      RQ2 abstract condition
logs/variance/                      k=5 repeated sampling
logs/Mech-Design/                   RQ4 original runs
logs/Mech-Design-nsw/               RQ4 rerun of the 342 scenarios whose mechanism prompt changed
```

**Scoring conventions used throughout.** Answers are re-parsed from the model output
(`analysis/reviewer_response/extract_choices.py`) rather than taken from the scores
stored in the logs. The stored scores mis-parse decorated answers such as Claude's
`**ANSWER:**`. An unparsed or refused answer counts as wrong, and the denominator is
every scenario in the game. Targets come from the current `gt-harmbench-cleaned.csv`,
which uses the own-minimum Nash social welfare.

## Headline numbers

| Claim | Value | Produced by | Needs |
|---|---|---|---|
| Socially optimal outcome rate, 15 models × 1,535 (abstract, §3.1, conclusion) | 61% (fail 39%) | `analysis.reviewer_response.emit_tables` → weighted average of Table 2 | logs |
| Mechanisms improve outcomes "by up to 31%" (abstract, intro, conclusion) | +30.7 pts utilitarian, mediator | `analysis.reviewer_response.rq4_figures` → `analysis/reviewer_response/figures/rq4/rq4_numbers.json` (`mechanisms.mediator.d_util`) | logs (or the committed JSON) |
| Evaluation awareness (§2) | 0.34% overall (9/2,666), no model > 2%, 10 models | `analysis/reviewer_response/eval_awareness_breakdown.py`, Sonnet 5 judge, as-judged | committed `artifacts/eval_awareness/judged_claude-sonnet-5.jsonl` |
| Game-type inter-annotator agreement (§2, App. E.5) | κ = 0.84, 26/30 | `scripts/validation/compute.py` | committed annotator files |
| Structural verifier pass rate (§2, App. E.7) | 1,530/1,535 (99.7%) | `analysis.reviewer_response.run_analysis` (Task 1, uses `is_valid_game` from `scripts/generation/fix_contextualizations.py`) | dataset only |

## Main text

| Element | Label | Produced by | Needs |
|---|---|---|---|
| Fig. 1, pipeline diagram | `fig:figure-1` | drawn by hand | — |
| Table 1, comparison with prior benchmarks | `tab:related_work` | literature review, no code | — |
| Fig. 2, example scenario (id 1094) | `fig:figure_2` | row `id == 1094` of `data/gt-harmbench-cleaned.csv`; model answers from the baseline logs | dataset, logs |
| Dataset construction: 2,491 candidates → 2,009 → 1,535 (§2) | — | `scripts/generation/` stages 1–3 (see README, *Generating Dataset*); prompt in `scripts/generation/generation_prompt.md` | API keys |
| **Table 2**, utilitarian accuracy per model × game | `tab:accuracy_heatmap_transposed` | `analysis.reviewer_response.emit_tables` (LaTeX); `analysis.reviewer_response.reproduce_tables` compares against the Overleaf source | logs |
| SD column / "1.6 pts, 92% majority agreement" (§3.1) | in Table 2 | `analysis.reviewer_response.variance_analysis` | `logs/variance/` |
| Per-game failure rates in prose (PD 47%, Chicken 77%, BoS 47%, Coordination 80%) | §3.1 | columns of Table 2 | logs |
| **Fig. 3**, framing analysis (prosaic → payoffs shown) | `fig:gamify-variation` | numbers: `analysis.reviewer_response.gamified_comparison` (PD+Chicken weighted; Nash +7.7, util −10.6 pts). The plot itself was drawn outside this repo | baseline + `RQ2 - gamified` logs |
| **Fig. 4**, order effect on Coordination | `fig:ordering` | `analysis.reviewer_response.ordering_effect` (11 models, 180 scenarios, −10.5 pts) | `RQ2 - gamified` + `RQ2-coordination` logs |
| RQ2 abstract condition (§3.2 paragraph) | — | `analysis.reviewer_response.abstract_vs_narrative` | baseline + abstract logs |
| **Fig. 5**, reasoning categories by outcome (RQ3) | `fig:reasoning-figure` | `analysis/reasoning/` pipeline: `extract.py` → `classifier.py` / `classifier_batch.py` → `plot_outcome_comparison.py`. See `analysis/reasoning/REASONING.md` | logs, API key |
| **Fig. 6**, mechanism effect, overall and per model (RQ4) | `fig:MD-figure4` | `analysis.reviewer_response.mech_rerun_merge --extract`, then `analysis.reviewer_response.rq4_figures` (writes `MD-Figure6.png`, `MD-Figure7.png`) | Mech-Design + Mech-Design-nsw logs |
| RQ4 mechanism prompts | — | `scripts/generation/generate_mechanism_*.py` (`--welfare nsw` for the camera-ready recommendation, `raw` reproduces `data/mech_variants/`); rerun: `scripts/experiments/mech_nsw_rerun.sh` | API keys |

## Appendix

| Element | Label | Produced by | Needs |
|---|---|---|---|
| Table 3, per-game counts | `tab:per_game_stats` | `data/gt-harmbench-cleaned.csv`, `formal_game` value counts (490 / 379 / 317 / 180 / 141 / 28) | dataset |
| Table 4, risk-domain coverage | `tab:domain_coverage_appendix` | `Domain` value counts of `data/gt-harmbench-cleaned.csv` vs `data/generation/mit.csv` | dataset |
| Fig. 7, cardinal-payoff robustness (App. E.4.1) | `fig:cardinal` | `analysis.reviewer_response.e4_cardinal_robustness` (seeded, 1,000 redraws) | dataset + baseline logs (or committed `analysis/reviewer_response/data/model_choices.csv`) |
| Symmetry corrections, 14 + 9 scenarios (App. E.4.1) | — | `analysis/reviewer_response/apply_mirror_patch.py`, then `apply_bos_tie_patch.py` | dataset |
| Nash social welfare definition (own-minimum disagreement point) | — | `game/metrics.py::nash_social_payoff`; targets rewritten by `analysis.reviewer_response.regenerate_nsw_targets` | dataset |
| Table 5, evaluation awareness per model × game (App. E.6) | `tab:cot-counts` | `analysis/reviewer_response/eval_awareness_breakdown.py`; judge: `eval_awareness_judge.py` (Haiku) / `eval_awareness_judge2.py` (Sonnet 5) | committed `artifacts/eval_awareness/` |
| Table 6, structural verification per game (App. E.7) | — | `is_valid_game` in `scripts/generation/fix_contextualizations.py` | dataset |
| Fig. 8, dataset-generation workflow | `fig:workflow-generation` | drawn by hand | — |
| Tables 7–9, Rawlsian / Nash-equilibrium / NSW accuracy | `tab:accuracy_heatmap_2`, `_3`, `_nsw` | `analysis.reviewer_response.emit_tables` | logs |
| Fig. 10, cross-play grid | `fig:cross-play` | `analysis.reviewer_response.cross_play` (cross 0.566 vs self 0.627) | logs |
| Table 10, repeated sampling (k=5) | `tab:variance` | `analysis.reviewer_response.variance_analysis` | `logs/variance/` |
| Table 11, capability checks (game classification, Nash detection) | `tab:due_diligence` | `eval/due_diligence.py` (`scripts/experiments/due_diligence.sh`) | logs |
| Figs. 11–12, reasoning categories by game / by model | `fig:reasoning-by-game`, `fig:reasoning-by-model` | `analysis/reasoning/plot_results.py` | classified traces |
| Table 12, reasoning categories | `tab:reasoning_categories` | category definitions in `analysis/reasoning/classifier.py` | — |
| Figs. 13–16, mechanism variants and heatmaps | `fig:app-MD1` … `fig:app-MD4` | `analysis.reviewer_response.rq4_figures` | as Fig. 6 |
| Figs. 17–18, joint-choice distributions | `fig:app:game-distribution.`, `fig:app:game-distribution-2` | `analysis.reviewer_response.choice_distribution` | logs |
| Table 13, narrative → abstract shift per game | `tab:abstract-shift` | `analysis.reviewer_response.abstract_vs_narrative` | baseline + abstract logs |

## Re-running evaluations from scratch

The rows above re-score existing logs. To regenerate the logs themselves, use the
batch runners in `scripts/experiments/` (see README, *Reproducing the paper*).
Several models from the paper are no longer served (Gemini 3 Pro, Grok 4.1 Fast,
and some preview endpoints), so a fresh run will not match every row exactly.
