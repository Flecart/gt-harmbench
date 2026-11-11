#!/bin/bash
# RQ4 — mechanism design.
#
# Evaluates a model against the mechanism-augmented datasets in data/mech_variants/:
# five mechanism families x four framing strengths = 20 runs per model.
#
# Usage:
#   ./scripts/experiments/mechanisms.sh                       # all 20 variants
#   ./scripts/experiments/mechanisms.sh mediator              # one family
#   ./scripts/experiments/mechanisms.sh mediator-v1_initial   # one variant
#
# Set MODEL / TEMPERATURE / LIMIT to override the defaults below. Start with
# LIMIT=20 to sanity-check cost before launching the full sweep — 20 variants
# x 1,535 scenarios x 2 player perspectives is a lot of API calls.

set -euo pipefail

MODEL="${MODEL:-openai/gpt-5.1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LOG_DIR="${LOG_DIR:-logs/mechanisms}"
LIMIT="${LIMIT:-}"

pattern="${1:-}"
mapfile -t datasets < <(ls data/mech_variants/*.csv | grep -- "${pattern}" | sort)

if [ ${#datasets[@]} -eq 0 ]; then
    echo "No mechanism datasets matched '${pattern}'." >&2
    echo "Available families: contracts mediator message payments penalties" >&2
    exit 1
fi

echo "Model:    ${MODEL}"
echo "Variants: ${#datasets[@]}"
echo "Logs:     ${LOG_DIR}"
echo

model_clean=${MODEL##*/}

for dataset in "${datasets[@]}"; do
    variant=$(basename "$dataset" .csv)
    echo "==> ${variant}"
    uv run python3 -m eval.eval \
        --model-name "$MODEL" \
        --dataset "$dataset" \
        --times 1 \
        --temperature "$TEMPERATURE" \
        --experiment-name "mech-${variant}-${model_clean}" \
        --log-dir "$LOG_DIR" \
        ${LIMIT:+--limit "$LIMIT"}
done

echo
echo "Done. Compare against the RQ1 baseline in logs/standard/ to measure the"
echo "mechanism effect; see analysis/mechanisms/ for the figure code."
