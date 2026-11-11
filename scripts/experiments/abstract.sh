#!/bin/bash
# RQ2 framing study: abstract-generic condition (no story, generic action labels).
# Uses --abstract flag which swaps system prompt to "game theory scenario" instead of "high stakes scenario".
# Compare results against existing GT-HarmBench logs in logs/GDrive/ (same models, same temperature).
#
# Prerequisites: generate the full abstract dataset first (only once):
#   uv run python scripts/generation/generate_abstract.py --full
# Output: data/gt-harmbench-abstract-full.csv (1535 rows, matches GT-HarmBench distribution exactly)
#
# Analysis after all runs complete:
#   uv run python scripts/analysis/rq2_framing.py \
#       --abstract-dir logs/abstract \
#       --narrative-dir logs/baseline

models=(
    "openrouter/anthropic/claude-4.5-opus"
    "openrouter/anthropic/claude-sonnet-4.5"
    "openai/gpt-5.2"
    "openai/gpt-5.1"
    "openai/gpt-5-mini-2025-08-07"
    "openai/gpt-5-nano-2025-08-07"
    "openai/gpt-4o-2024-08-06"
    "openrouter/x-ai/grok-4.1-fast"
    "openrouter/google/gemini-3-pro-preview"
    "openrouter/google/gemini-3-flash-preview"
    "openrouter/meta-llama/llama-3.3-70b-instruct"
    "openrouter/meta-llama/llama-3.2-3b-instruct"
    "openrouter/qwen/qwen3-30b-a3b"
    "openrouter/Qwen/Qwen3-8B"
    "openrouter/deepseek/deepseek-v3.2"
)

max_jobs=4

wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 0.1
    done
}

run_eval() {
    local model=$1
    model_clean=${model##*/}
    echo "Running: $model"
    uv run python3 -m eval.eval \
        --model-name "$model" \
        --dataset data/gt-harmbench-abstract-full.csv \
        --log-dir logs/abstract \
        --times 1 \
        --temperature 1.0 \
        --abstract \
        --experiment-name "rq2-abstract-${model_clean}"
}

for model in "${models[@]}"; do
    wait_for_slot
    run_eval "$model" &
done

wait
echo "Done."
