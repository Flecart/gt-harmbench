#!/bin/bash
# RQ3 — classify the reasoning traces behind a model's choices.
#
# Runs an LLM judge over the reasoning text in an inspect_ai .eval log and
# labels each trace against the taxonomy in analysis/reasoning/classifier.py.
#
# Usage:
#   ./scripts/experiments/analyze_reasoning.sh <log-file.eval> [output.csv]
#
# With no arguments it picks the most recent log under logs/. Override the
# judge with JUDGE_MODEL, and the sample count with LIMIT (default 20 — the
# judge costs one API call per trace, so raise it deliberately).

set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4o-mini}"
LIMIT="${LIMIT:-20}"
SEED="${SEED:-123}"

log_file="${1:-}"
if [ -z "$log_file" ]; then
    log_file=$( { find logs -name '*.eval' -type f 2>/dev/null || true; } | xargs -r ls -t 2>/dev/null | head -1 || true)
    if [ -z "$log_file" ]; then
        echo "No .eval logs found under logs/." >&2
        echo "Run an evaluation first, e.g. ./scripts/experiments/standard.sh" >&2
        echo "Usage: $0 <log-file.eval> [output.csv]" >&2
        exit 1
    fi
    echo "No log given; using most recent: $log_file"
fi

output="${2:-reasoning_analysis.csv}"

uv run python analysis/reasoning/reasoning_analysis.py \
    "$log_file" \
    --judge-model "$JUDGE_MODEL" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --output "$output"

echo "Wrote $output"
