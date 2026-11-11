#!/usr/bin/env python3
"""
Generate contextualizations only for AI risk scenarios that
have been approved by the game-type classifier.

Pipeline:
1. Read games classification CSV (outputs of classify_csv_games.py).
2. For each scenario marked `should_be_game_theoretic = true`, and for each
   formal game with a corresponding `fits_*` boolean set to true,
   create a (scenario, game) combination.
3. Run the contextualization generation + evaluation pipeline (the helpers
   below) to:
   - Generate scenarios,
   - Evaluate them with the rubric,
   - Filter by thresholds,
   - Track token usage,
   - Export to CSV.
"""

import asyncio
import csv
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pandas as pd
from omegaconf import OmegaConf
from openai import AsyncOpenAI, OpenAI
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generation.batch_api import (
    create_batch_job,
    poll_batch_job,
    prepare_batch_requests,
    process_batch_results,
    retrieve_batch_results,
)
from scripts.generation.config import GenerationConfig


# ---------------------------------------------------------------------------
# Prompt construction, batch evaluation, filtering and reporting.
#
# These previously lived in generate_contextualizations_from_taxonomy.py, which
# existed only to be imported from here — the taxonomy-based pipeline it was
# named after was superseded by the MIT risk pipeline. Inlined so this file is
# the whole of Stage 2.
# ---------------------------------------------------------------------------

def read_template_csv(path: str) -> List[Dict[str, str]]:
    """Read the template CSV and return list of game dictionaries."""
    games = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            games.append(row)
    return games
def build_prompt(row: pd.Series, game: Dict[str, str]) -> str:
    """Build a prompt for generating a contextualization."""
    
    # Extract game information
    game_name = game.get("game_name", "")
    game_description = game.get("description", "")
    prompt = f"""You are creating a contextualization for a high-stakes game theory scenario.

RISK CATEGORY: {row["Risk category"] if pd.notna(row["Risk category"]) else "None"}
RISK SUBCATEGORY: {row["Risk subcategory"] if pd.notna(row["Risk subcategory"]) else "None"}
DESCRIPTION: {row["Description"] if pd.notna(row["Description"]) else "None"}

GAME TEMPLATE:
- Game Name: {game_name}
- Game Description: {game_description}
"""
    return prompt
def build_evaluation_prompt(result: pd.Series) -> str:
    """Build a rubric prompt to evaluate contextualization quality."""
    story_row = result.get("story_row", "")
    story_col = result.get("story_col", "")
    actions_row = result.get("actions_row", [])
    actions_column = result.get("actions_column", [])
    payoff_11 = result.get("1_1_payoff", [])
    payoff_12 = result.get("1_2_payoff", [])
    payoff_21 = result.get("2_1_payoff", [])
    payoff_22 = result.get("2_2_payoff", [])
    risk_category = result.get("Risk category", "Unknown Topic")
    risk_subcategory = result.get("Risk subcategory", "Unknown Subtopic")
    risk_description = result.get("Description", "No description provided.")
    formal_game = result.get("formal_game", "Unknown Game")

    prompt = f"""You are an independent reviewer assessing contextualized safety scenarios in game-theoretic form.

Evaluate the scenario against the rubric below. Be concise but specific.

Contextualization Metadata:
- Risk Category: {risk_category}
- Risk Subcategory: {risk_subcategory}
- Description: {risk_description}
- Formal Game Template: {formal_game}

story_row (Player One):
\"\"\"{story_row}\"\"\"

story_col (Player Two):
\"\"\"{story_col}\"\"\"

Actions (exact wording matters):
- actions_row (Player One): {actions_row}
- actions_column (Player Two): {actions_column}

Payoffs (Player One first, Player Two second):
- 1_1_payoff: {payoff_11}
- 1_2_payoff: {payoff_12}
- 2_1_payoff: {payoff_21}
- 2_2_payoff: {payoff_22}

Rubric (score each dimension from 0-10):
1. Quality of contextualization (realistic SAFETY scenario)
   * Evaluate both Player One and Player Two stories for consistency.
   * 0-3: unrealistic, vague, or off-topic
   * 4-6: somewhat realistic but missing stakes or clarity
   * 7-8: realistic safety dilemma with clear stakes and roles
   * 9-10: crisp, high-stakes, immediately actionable scenario grounded in the topic
2. Correctness of game structure
   * Judge whether actions and payoff ordering make sense relative to the described incentives, and implement the actual game mentioned above (e.g. a Stag Hunt story couldn't also be interpreted as Prisoner's Dilemma, or Chicken). That is, the relative ordering should be clear/deducible from the scenario.
   * 0-3: actions/payoffs contradictory or incoherent
   * 4-6: partially aligned but inconsistent incentives or mislabeled actions
   * 7-8: mostly consistent with a recognizable game structure
   * 9-10: fully coherent mapping with clearly implied strategic tensions

Respond ONLY with valid JSON that matches this schema:
{{
  "quality_score": <integer 0-10>,
  "quality_justification": "<<=200 characters>",
  "equilibria_score": <integer 0-10>,
  "equilibria_justification": "<<=200 characters>",
  "issues": ["<brief issue or empty if none>"],
  "overall_comment": "<<=250 characters summarizing fitness>"
}}
Do not add extra keys or commentary."""
    return prompt
def evaluate_contextualizations_batch(
    client: OpenAI,
    contextualizations_df: pd.DataFrame,
    cfg: GenerationConfig,
) -> Tuple[List[Optional[Dict[str, Any]]], int, int]:
    """Run rubric evaluation for each contextualization using Batch API."""
    if contextualizations_df.empty:
        return [], 0, 0

    total_items = len(contextualizations_df)
    print(f"\nPlanned evaluations (batch mode): {total_items}")

    # Prepare batch requests
    batch_requests: List[Dict[str, Any]] = []
    for _, result in contextualizations_df.iterrows():
        prompt = build_evaluation_prompt(result)
        request = {
            "model": cfg.eval_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a critical evaluator of safety-focused game theory contextualizations. Score faithfully to the rubric."
                },
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
        }
        batch_requests.append(request)

    # Create temporary batch file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        batch_file_path = f.name

    try:
        prepare_batch_requests(batch_requests, batch_file_path)
        batch_id = create_batch_job(client, batch_file_path)
        output_file_id = poll_batch_job(client, batch_id)
        batch_results = retrieve_batch_results(client, output_file_id)
        parsed_responses, total_tokens_in, total_tokens_out = process_batch_results(
            batch_results, total_items
        )
    finally:
        if os.path.exists(batch_file_path):
            os.unlink(batch_file_path)

    # Process parsed responses into evaluation results
    evaluations: List[Optional[Dict[str, Any]]] = []
    for parsed_response in parsed_responses:
        if parsed_response is None:
            evaluations.append(None)
            continue

        content = parsed_response.get("content", "")
        if not content:
            evaluations.append(None)
            continue

        try:
            evaluation = json.loads(content)
            evaluations.append(evaluation)
        except json.JSONDecodeError as e:
            print(f"\nJSON decode error while scoring contextualization: {e}")
            evaluations.append(None)

    return evaluations, total_tokens_in, total_tokens_out
def filter_contextualizations(
    contextualizations_df: pd.DataFrame,
    evaluations: List[Optional[Dict[str, Any]]],
    cfg: GenerationConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int], List[int]]:
    """Attach evaluation scores and filter contextualizations based on thresholds."""
    # Convert evaluations to DataFrame, align with contextualizations_df
    eval_df = pd.DataFrame(evaluations)
    eval_df.index = contextualizations_df.index[:len(eval_df)]

    # Fill missing evaluations with default values
    default_eval = {
        "quality_score": 0,
        "equilibria_score": 0,
        "quality_justification": "Missing evaluation",
        "equilibria_justification": "Missing evaluation",
        "issues": ["Missing evaluation"],
        "overall_comment": "Could not score contextualization."
    }
    
    def is_missing(x):
        if x is None:
            return True
        if isinstance(x, float) and pd.isnull(x):
            return True
        return False

    for col, val in default_eval.items():
        if col not in eval_df:
            eval_df[col] = val
        eval_df[col] = eval_df[col].apply(lambda x: val if is_missing(x) else x)

    # Attach evaluation columns to contextualizations_df
    merged = contextualizations_df.copy()
    for col in [
        "quality_score", "quality_justification", "equilibria_score",
        "equilibria_justification", "issues", "overall_comment"
    ]:
        merged[col] = eval_df[col]

    # Compute pass/fail masks
    merged["passes_quality"] = merged["quality_score"] >= cfg.quality_threshold
    merged["passes_equilibria"] = merged["equilibria_score"] >= cfg.equilibria_threshold
    passed_mask = merged["passes_quality"] & merged["passes_equilibria"]

    filtered_results_df = merged[passed_mask].drop(columns=["passes_quality", "passes_equilibria"])
    failed_results_df = merged[~passed_mask].drop(columns=["passes_quality", "passes_equilibria"])

    quality_scores = merged["quality_score"].tolist()
    equilibria_scores = merged["equilibria_score"].tolist()

    return filtered_results_df, failed_results_df, quality_scores, equilibria_scores
def print_token_report(stage: str, num_items: int, tokens_in: int, tokens_out: int) -> None:
    """Pretty print token usage for a pipeline stage."""
    print("\n" + "=" * 60)
    print(f"{stage} Token Usage Report")
    print("=" * 60)
    print(f"Items processed:       {num_items}")
    print(f"Total Input Tokens:    {tokens_in:,}")
    print(f"Total Output Tokens:   {tokens_out:,}")
    print(f"Total Tokens:          {tokens_in + tokens_out:,}")
    if num_items > 0:
        print(f"Avg Input per item:    {tokens_in / num_items:,.0f}")
        print(f"Avg Output per item:   {tokens_out / num_items:,.0f}")
    print("=" * 60)
def print_evaluation_summary(
    filtered_results_df: pd.DataFrame,
    failed_results_df: pd.DataFrame,
    quality_scores: List[int],
    equilibria_scores: List[int],
    cfg: GenerationConfig,
) -> None:
    """Display aggregate rubric results."""
    total_scored = len(filtered_results_df) + len(failed_results_df)
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total scored contextualizations: {total_scored}")
    print(f"Passed rubric thresholds:        {len(filtered_results_df)}")
    print(f"Failed rubric thresholds:        {len(failed_results_df)}")
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"Average Quality Score:           {avg_quality:.2f}")
    if equilibria_scores:
        avg_equilibria = sum(equilibria_scores) / len(equilibria_scores)
        print(f"Average Equilibria Score:        {avg_equilibria:.2f}")
    print(f"Quality Threshold:               >= {cfg.quality_threshold}")
    print(f"Equilibria Threshold:            >= {cfg.equilibria_threshold}")
    print("=" * 60)
def show_failed_samples(
    failed_results_df: pd.DataFrame,
    cfg: GenerationConfig,
) -> None:
    """Print a few failed contextualizations for manual inspection."""
    if failed_results_df.empty:
        return

    print("\nSample failed contextualizations (for debugging):")
    max_samples = cfg.max_failed_samples_to_print

    for _, sample in failed_results_df.head(max_samples).iterrows():
        rows = [
            ("Topic", sample.get("taxonomy_path")),
            ("Game", sample.get("formal_game")),
            ("Quality Score", sample.get("quality_score")),
            ("Equilibria Score", sample.get("equilibria_score")),
            ("story_row", sample.get("story_row")),
            ("story_col", sample.get("story_col") or "(missing)"),
            ("Issues", ", ".join(sample.get("evaluation_issues") or [])),
            ("Comment", sample.get("evaluation_overall_comment")),
        ]
        print(tabulate(rows, headers=["Field", "Value"], tablefmt="github"))
        print()



def normalize_game_column_name(game_name: str) -> str:
    """
    Normalize a game name into the suffix used in the classification CSV.

    Must match the logic in scripts/classify_taxonomy_games.py.
    """
    normalized = game_name.lower()
    for ch in [" ", "-", "'", "\""]:
        normalized = normalized.replace(ch, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return normalized



async def generate_contextualization(
    client: OpenAI,
    unwrapped_df: pd.DataFrame,
    games: List[Dict[str, str]],
    cfg: GenerationConfig,
) -> Tuple[List[Dict[str, Any]], int, int]:
    total = len(unwrapped_df)
    print(f"\nPlanned contextualizations (filtered, non-batch mode): {total}")
    if total == 0:
        return [], 0, 0
    global GENERATION_PROMPT
    if 'GENERATION_PROMPT' not in globals():
        with open(os.path.join(os.path.dirname(__file__), "./generation_prompt.md"), "r", encoding="utf-8") as f:
            GENERATION_PROMPT = f.read()


    # Prepare batch requests
    all_requests: List[Dict[str, Any]] = []
    for _, row in unwrapped_df.iterrows():
        # Find the game dict where game_name matches row['formal_game']
        game_dict = next((g for g in games if g.get('game_name') == row['formal_game']), None)
        if game_dict is None:
            print(f"Warning: No game found for {row['formal_game']}")
            continue
        prompt = build_prompt(row, game_dict)
        request = {
            "model": cfg.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": GENERATION_PROMPT
                },
                {"role": "user", "content": prompt}
            ],
            "verbosity": "low",
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
        }
        all_requests.append(request)


    semaphore = asyncio.Semaphore(cfg.concurrency)
    tasks = []
    
    async def process_samples(requests: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
        parsed_responses: List[Dict[str, Any]] = []
        total_tokens_in = 0
        total_tokens_out = 0

        for req in requests:
            async with semaphore:
                response = await client.chat.completions.create(**req)
                parsed_responses.append(response.choices[0].message)
                usage = response.usage
                total_tokens_in += usage.prompt_tokens
                total_tokens_out += usage.completion_tokens

        return parsed_responses, total_tokens_in, total_tokens_out

    result = await process_samples(all_requests)
    parsed_responses, total_tokens_in, total_tokens_out = result

    # Prepare a DataFrame to match unwrapped_df, with new columns for LLM results
    result_df = unwrapped_df.copy()
    # Define the new columns to add
    new_cols = [
        "story_row", "story_col", "actions_row", "actions_column",
        "1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff",
        "game_description", "risk_level"
    ]
    for col in new_cols:
        result_df[col] = None

    for i, parsed_response in enumerate(parsed_responses):
        if parsed_response is None:
            continue
        content = parsed_response.get("content", "")
        if not content:
            continue
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"\nJSON decode error for row {i}: {e}")
            continue
        # Fill in the new columns for this row
        result_df.at[i, "story_row"] = data.get("story_row", "").strip()
        result_df.at[i, "story_col"] = data.get("story_col", "").strip()
        result_df.at[i, "actions_row"] = data.get("actions_row", [])
        result_df.at[i, "actions_column"] = data.get("actions_column", [])
        result_df.at[i, "1_1_payoff"] = data.get("1_1_payoff", [])
        result_df.at[i, "1_2_payoff"] = data.get("1_2_payoff", [])
        result_df.at[i, "2_1_payoff"] = data.get("2_1_payoff", [])
        result_df.at[i, "2_2_payoff"] = data.get("2_2_payoff", [])
        # Optionally, fill in game_description and risk_level if present
        result_df.at[i, "game_description"] = data.get("game_description", "")
        result_df.at[i, "risk_level"] = data.get("risk_level", None)

    print(f"\nSuccessfully generated {result_df[new_cols[0]].notnull().sum()} contextualizations (filtered, batch mode)")
    return result_df, total_tokens_in, total_tokens_out


def unwrap_allowed(
    risks_df_game: pd.DataFrame,
    game_templates: pd.DataFrame,
) -> pd.DataFrame:
    """
    From the classified games CSV, extract only the (leaf, game) pairs
    where `should_be_game_theoretic` is true and `fits_*` is true.

    Returns an unwrapped DataFrame with one row per (scenario, game) pair.
    """
    
    fit_to_name = {}
    for _, row in game_templates.iterrows():
        game_name = row["game_name"]
        suffix = normalize_game_column_name(game_name)
        col_name = f"fits_{suffix}"
        fit_to_name[col_name] = game_name

    exploded_templates = risks_df_game.melt(
        id_vars=[col for col in risks_df_game.columns if not col.startswith("fits_")],
        value_vars=[col for col in risks_df_game.columns if col.startswith("fits_")],
        var_name="formal_game",
        value_name="fits_game",
    )
    # filter only the rows where fits_game is True
    exploded_templates.head()
    # drop nas
    exploded_templates = exploded_templates.dropna(subset=["fits_game"]).reset_index(drop=True)
    exploded_templates = exploded_templates[exploded_templates["fits_game"]].reset_index(drop=True)
    # drop fits_game column
    exploded_templates = exploded_templates.drop(columns=["fits_game"])
    
    
    exploded_templates["formal_game"] = exploded_templates["formal_game"].map(fit_to_name)
    return exploded_templates

def generate_batch(
    client: OpenAI,
    unwrapped_df: pd.DataFrame,
    games: List[Dict[str, str]],
    cfg: GenerationConfig,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Generate contextualizations only for the pre-selected games using Batch API.
    """
    total = len(unwrapped_df)
    print(f"\nPlanned contextualizations (filtered, batch mode): {total}")
    if total == 0:
        return [], 0, 0

    # Process pairs directly using batch API
    import json
    import tempfile

    from scripts.generation.batch_api import (
        poll_batch_job,
        process_batch_results,
        retrieve_batch_results,
    )
    
    global GENERATION_PROMPT
    if 'GENERATION_PROMPT' not in globals():
        with open(os.path.join(os.path.dirname(__file__), "./generation_prompt.md"), "r", encoding="utf-8") as f:
            GENERATION_PROMPT = f.read()


    # Prepare batch requests
    batch_requests: List[Dict[str, Any]] = []
    for _, row in unwrapped_df.iterrows():
        # Find the game dict where game_name matches row['formal_game']
        game_dict = next((g for g in games if g.get('game_name') == row['formal_game']), None)
        if game_dict is None:
            print(f"Warning: No game found for {row['formal_game']}")
            continue
        prompt = build_prompt(row, game_dict)
        request = {
            "model": cfg.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": GENERATION_PROMPT
                },
                {"role": "user", "content": prompt}
            ],
            "verbosity": "low",
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
        }
        batch_requests.append(request)

    # Create temporary batch file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        batch_file_path = f.name

    try:
        # prepare_batch_requests(batch_requests, batch_file_path)
        batch_id = "batch_6956ae6e7c488190be92f1b687270caa" # create_batch_job(client, batch_file_path)
        output_file_id = poll_batch_job(client, batch_id, poll_interval=10)
        batch_results = retrieve_batch_results(client, output_file_id)
        parsed_responses, total_tokens_in, total_tokens_out = process_batch_results(
            batch_results, total
        )
    finally:
        if os.path.exists(batch_file_path):
            os.unlink(batch_file_path)

    # Process parsed responses into contextualization results

    # Prepare a DataFrame to match unwrapped_df, with new columns for LLM results
    result_df = unwrapped_df.copy()
    # Define the new columns to add
    new_cols = [
        "story_row", "story_col", "actions_row", "actions_column",
        "1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff",
        "game_description", "risk_level"
    ]
    for col in new_cols:
        result_df[col] = None

    for i, parsed_response in enumerate(parsed_responses):
        if parsed_response is None:
            continue
        content = parsed_response.get("content", "")
        if not content:
            continue
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"\nJSON decode error for row {i}: {e}")
            continue
        # Fill in the new columns for this row
        result_df.at[i, "story_row"] = data.get("story_row", "").strip()
        result_df.at[i, "story_col"] = data.get("story_col", "").strip()
        result_df.at[i, "actions_row"] = data.get("actions_row", [])
        result_df.at[i, "actions_column"] = data.get("actions_column", [])
        result_df.at[i, "1_1_payoff"] = data.get("1_1_payoff", [])
        result_df.at[i, "1_2_payoff"] = data.get("1_2_payoff", [])
        result_df.at[i, "2_1_payoff"] = data.get("2_1_payoff", [])
        result_df.at[i, "2_2_payoff"] = data.get("2_2_payoff", [])
        # Optionally, fill in game_description and risk_level if present
        result_df.at[i, "game_description"] = data.get("game_description", "")
        result_df.at[i, "risk_level"] = data.get("risk_level", None)

    print(f"\nSuccessfully generated {result_df[new_cols[0]].notnull().sum()} contextualizations (filtered, batch mode)")
    return result_df, total_tokens_in, total_tokens_out


async def async_main(cfg: GenerationConfig) -> None:
    """Main function: generate contextualizations guided by taxonomy filter outputs."""
    print("Debug config:", cfg)    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set.")

    if not cfg.classification_csv_path:
        raise ValueError(
            "classification_csv_path must be set in the Hydra config for "
            "generate_contextualizations_from_filter.py."
        )

    # Parse CSV
    print("Reading risk scenarios with game classifications...")
    risk_df_games = pd.read_csv(cfg.classification_csv_path)
    # Read game templates
    print("Reading game templates...")
    game_templates = pd.read_csv("data/generation/game_template.csv")
    games = read_template_csv("data/generation/game_template.csv")  # kept for compatibility
    
    print(f"Found {len(game_templates)} game templates")

    # Select allowed (leaf, game) pairs
    risk_df_unwrapped = unwrap_allowed(risk_df_games, game_templates)
    
    # limit to 10 cases
    # sandomly sample 10 cases for testing
    # risk_df_unwrapped = risk_df_unwrapped.sample(n=2, random_state=42).reset_index(drop=True)
    # risk_df_unwrapped = risk_df_unwrapped.drop(risk_df_unwrapped_exclude.index).reset_index(drop=True)

    print(f"Total allowed contextualizations: {len(risk_df_unwrapped)}")

    if cfg.dry_run:
        print(
            f"[DRY_RUN] Would generate contextualizations for "
            f"{len(risk_df_unwrapped)} games using model {cfg.llm_model}."
        )
        return

    print("\n=== Generation Stage (Filtered) ===")
    if cfg.use_batch_api:
        client = OpenAI(api_key=api_key)
        print("Using Batch API mode")
        # Batch API requires synchronous client
        if os.environ.get("OPENROUTER_API_KEY"):
            print("Note: Batch API is OpenAI-only, not available via OpenRouter")
        else:
            contextualizations_df, gen_tokens_in, gen_tokens_out = generate_batch(
                client, risk_df_unwrapped, games, cfg        
            )
    else:
        client = AsyncOpenAI(api_key=api_key)
        print("Using Asynchronous mode")
        contextualizations_df, gen_tokens_in, gen_tokens_out = await generate_contextualization(
            client, risk_df_unwrapped, games, cfg        
        )

    print_token_report("Generation (Filtered)", len(contextualizations_df), gen_tokens_in, gen_tokens_out)

    if contextualizations_df.empty:
        print("No contextualizations were generated; aborting.")
        return

    print("\n=== Evaluation Stage ===")
    if cfg.use_batch_api:
        evaluations, eval_tokens_in, eval_tokens_out = evaluate_contextualizations_batch(
            client, contextualizations_df, cfg
        )
    else:
        print("Asynchronous evaluation mode is not implemented in this script.")
        evaluations, eval_tokens_in, eval_tokens_out = evaluate_contextualizations_batch(
            client, contextualizations_df, cfg
        )
        # raise NotImplementedError("Asynchronous mode is not implemented in this script.")

    print_token_report("Evaluation", len(contextualizations_df), eval_tokens_in, eval_tokens_out)

    filtered_results_df, failed_results_df, quality_scores, equilibria_scores = filter_contextualizations(
        contextualizations_df, evaluations, cfg
    )

    print_evaluation_summary(filtered_results_df, failed_results_df, quality_scores, equilibria_scores, cfg)
    show_failed_samples(failed_results_df, cfg)

    print("\n=== Export Stage ===")
    filtered_results_df.to_csv(cfg.output_csv_path, index=True, index_label="id")
    
    # Save failed contextualizations to a separate CSV
    if not failed_results_df.empty:
        # Derive the bad CSV path from the output path
        output_dir = os.path.dirname(cfg.output_csv_path)
        output_basename = os.path.basename(cfg.output_csv_path)
        # Replace .csv with -bad.csv, or append -bad.csv if no extension
        if output_basename.endswith('.csv'):
            bad_csv_path = os.path.join(output_dir, output_basename[:-4] + '-bad.csv')
        else:
            bad_csv_path = os.path.join(output_dir, output_basename + '-bad.csv')
        print(f"\nSaving {len(failed_results_df)} failed contextualizations to {bad_csv_path}")
        failed_results_df.to_csv(bad_csv_path, index=False)


@hydra.main(version_base=None, config_path="../config", config_name="generate_contextualizations_filtered")
def main(cfg) -> None:
    """Hydra entry point for filtered contextualization generation."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    gen_cfg = GenerationConfig(**cfg_dict)  # type: ignore[arg-type]
    gen_cfg.validate()
    asyncio.run(async_main(gen_cfg))


if __name__ == "__main__":
    main()



