---
language:
- en
license: cc-by-4.0
task_categories:
- question-answering
- text-classification
tags:
- game-theory
- ai-safety
- benchmark
- strategic-reasoning
- nash-equilibrium
- social-welfare
pretty_name: GT-HarmBench
size_categories:
- 1K<n<10K
---

# GT-HarmBench

**GT-HarmBench** is a game-theoretic AI safety benchmark that evaluates whether large language models can reason strategically in realistic, AI-risk–grounded scenarios.

Each scenario presents two players with a 2×2 payoff matrix embedded in a first-person narrative drawn from real AI risk contexts.
Models are evaluated on their ability to:

- identify and play **Nash equilibria** (individual rationality),
- select actions that maximise **utilitarian welfare** (sum of payoffs),
- select actions that maximise **Rawlsian welfare** (min payoff, fairness),
- select actions that maximise **Nash social welfare** (product of each player's gain over their own minimum payoff).

## Dataset summary

| | |
|---|---|
| Total scenarios | 1,535 |
| Columns | 20 |
| MIT AI Risk Database–sourced | 1,535 |
| Game types | 6 |

### Game type distribution

| Game type | Count |
|---|---|
| Prisoner's Dilemma | 490 |
| Chicken | 379 |
| Stag hunt | 317 |
| Coordination | 180 |
| Bach or Stravinski | 141 |
| No conflict | 28 |

### Risk level distribution

Scenarios are rated on a 1–10 severity scale.

| Risk level | Count |
|---|---|
| 3 | 3 |
| 4 | 61 |
| 5 | 51 |
| 6 | 494 |
| 7 | 660 |
| 8 | 170 |
| 9 | 88 |
| 10 | 8 |

## Dataset structure

### Columns

| Column | Description |
|---|---|
| `id` | Integer row identifier. |
| `ev_id` | MIT AI Risk Database event ID. |
| `domain` | Top-level domain of the MIT AI Risk Repository taxonomy (7 domains). |
| `risk_category` | Top-level risk category from the MIT AI Risk Database. |
| `risk_subcategory` | Sub-category from the MIT AI Risk Database. |
| `description` | Original risk description from the MIT AI Risk Database. |
| `formal_game` | Canonical game type: one of Prisoner's Dilemma, Chicken, Stag hunt, Coordination, Bach or Stravinski, No conflict. |
| `story_row` | First-person narrative given to the row player describing the strategic situation. |
| `story_col` | First-person narrative given to the column player. |
| `actions_row` | Python-list string of the two action labels available to the row player. |
| `actions_column` | Python-list string of the two action labels available to the column player. |
| `1_1_payoff` | Payoff vector [row, col] when both players choose action 1. |
| `1_2_payoff` | Payoff vector [row, col] when row plays action 1 and column plays action 2. |
| `2_1_payoff` | Payoff vector [row, col] when row plays action 2 and column plays action 1. |
| `2_2_payoff` | Payoff vector [row, col] when both players choose action 2. |
| `risk_level` | Integer 1–10 severity rating of the underlying AI risk. |
| `target_nash_equilibria` | Pipe-separated pure Nash equilibria as (row_action, col_action) tuples. |
| `target_utility_maximizing` | Pipe-separated action profiles that maximise the sum of payoffs (utilitarian welfare). |
| `target_rawlsian` | Pipe-separated action profiles that maximise the minimum payoff (Rawlsian fairness). |
| `target_nash_social_welfare` | Pipe-separated action profiles that maximise Nash social welfare: the product of each player's payoff minus their own minimum payoff in the game (security level as disagreement point). |

### Payoff matrix convention

Each row in the dataset encodes a 2×2 game.
The four payoff columns (`1_1_payoff` … `2_2_payoff`) each contain a two-element list `[row_payoff, col_payoff]`.
The matrix layout is:

```
                 col action 1        col action 2
row action 1    1_1_payoff          1_2_payoff
row action 2    2_1_payoff          2_2_payoff
```

### Target columns

Target columns encode optimal action profiles as pipe-separated strings of Python tuples, e.g.:

```
('go bold', 'go bold')|('play safe', 'play safe')
```

Multiple profiles appear when ties exist. Each tuple is `(row_action, col_action)`.

## Data sources

1. **MIT AI Risk Database** (`ev_id` is non-null, 1,535 rows): risk descriptions from the MIT AI Risk Repository were classified as game-theoretic and then contextualized into first-person strategic narratives by an LLM pipeline.

Scenarios were filtered by quality (score ≥ 8/10) and equilibria consistency (score ≥ 8/10) before inclusion.
Matching Pennies scenarios (mixed-strategy only, no pure Nash) are excluded.

## Changelog

- **Camera-ready release (NeurIPS 2026, Oct 2026).** 1,535 scenarios, all anchored to an MIT AI Risk Repository entry; adds the `domain` column. Nash social welfare targets now use each player's own minimum payoff as the disagreement point. 24 payoff matrices were corrected so every game is exactly symmetric (targets unchanged except the Nash social welfare tie in two Battle of the Sexes scenarios).
- **v0-2009 (Jul 2026).** Earlier pre-release (2,009 scenarios), used in the first arXiv version; differs in the scenario set and the Nash social welfare targets. Available as tag `v0-2009`.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Jinesis/gt-harmbench", split="train")
print(ds[0])
```

### Running the benchmark

```bash
# Evaluate a model
uv run python3 -m eval.eval \
    --model-name openai/gpt-4o \
    --dataset data/gt-harmbench-hf.csv \
    --times 1 --temperature 1.0 --experiment-name my-eval
```

See the [GitHub repository](https://github.com/Flecart/gt-harmbench) for the full evaluation harness.

## Citation

```bibtex
@dataset{gt-harmbench,
  title  = {GT-HarmBench: Benchmarking AI Safety Risks Through the Lens of Game Theory},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Jinesis/gt-harmbench}
}
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
