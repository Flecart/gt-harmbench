#!/bin/bash
# Choice stability under repeated sampling — promised to reviewer 5XrX.
#
# Commitment as posted:
#   "k=5 samples per model-scenario across all 15 models, on a 300-scenario
#    subset stratified by game type (~50 per type)."
#
# The subset already exists: data/gt-harmbench-rq2-sample.csv is 278 scenarios,
# 50 per game type plus all 28 No Conflict (the entire population for that
# game), drawn from the clean 1,535 with seed 42. Nothing to generate.
#
# --times 5 is k=5. It reaches inspect_ai as epochs=5, so each scenario gets 5
# independent records (verified: 2 scenarios x 3 epochs = 6 records). Group by
# (id, epoch) to get the per-scenario flip probability the reply promises.
#
# BUDGET. Everything routes through OpenRouter ($100/month cap); there is no
# working OpenAI key. Re-running all 13 still-available models costs ~$192, so
# it splits across two months:
#     batch1   11 models   ~$62    this month
#     batch2   2 Claude   ~$130    after the cap resets (Opus alone is ~$82)
#
# Usage:
#   ./scripts/experiments/variance.sh --dry-run   # plan + cost, runs nothing
#   ./scripts/experiments/variance.sh probe       # 1 scenario/model, id check (~$0)
#   ./scripts/experiments/variance.sh batch1      # 11 models, k=5  (~$62)
#   ./scripts/experiments/variance.sh batch2      # Sonnet + Opus   (~$130)
#
# ALWAYS run `probe` after changing model ids, and check raw outputs rather than
# the summary table: eval.py writes API errors into the answer field, so a run
# can report status=success with 100% failed calls and plausible-looking metrics.
#
# Re-running any group SKIPS models that already have a log in LOG_DIR, so an
# interrupted run resumes instead of re-billing what already completed.

set -euo pipefail

DATASET="data/gt-harmbench-rq2-sample.csv"
LOG_DIR="${LOG_DIR:-logs/variance}"
K="${K:-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LIMIT="${LIMIT:-}"

# Everything routes through OpenRouter — there is no working OpenAI key.
# batch1 is ordered cheapest-first so an interrupted run still yields models.
# RETIRED — 2 of the paper's 15 models were withdrawn by their providers between
# the original evaluation (May 2026) and this re-run, and cannot be re-sampled:
#   x-ai/grok-4.1-fast          404 "deprecated, xAI recommends Grok 4.3"
#   google/gemini-3-pro-preview 404 "No endpoints found" (line moved to 3.1-pro)
# Repeated sampling therefore covers 13 of 15 models. Disclose this in the reply.
batch1_models=(
    "openrouter/meta-llama/llama-3.2-3b-instruct"
    "openrouter/meta-llama/llama-3.3-70b-instruct"
    "openrouter/qwen/qwen3-8b"
    "openrouter/openai/gpt-5-nano"
    "openrouter/qwen/qwen3-30b-a3b"
    "openrouter/openai/gpt-4o"
    "openrouter/deepseek/deepseek-v3.2"
    "openrouter/openai/gpt-5-mini"
    "openrouter/openai/gpt-5.2"
    "openrouter/openai/gpt-5.1"
    "openrouter/google/gemini-3-flash-preview"
)
# batch2 is ~$130 and cannot share a month's quota with batch1.
batch2_models=(
    "openrouter/anthropic/claude-sonnet-4.5"
    "openrouter/anthropic/claude-4.5-opus"
)

group="${1:-}"
case "$group" in
    --dry-run) ;;
    batch1)  models=("${batch1_models[@]}") ;;
    batch2)  models=("${batch2_models[@]}") ;;
    probe)   models=("${batch1_models[@]}"); LIMIT=1 ;;
    *)
        echo "Usage: $0 {--dry-run|probe|batch1|batch2}" >&2
        echo "  probe   1 scenario per model — checks every model id resolves (cents)" >&2
        echo "  batch1  11 non-Claude models, k=5   ~\$62   (fits this month's cap)" >&2
        echo "  batch2  Sonnet + Opus, k=5         ~\$130  (run after the cap resets)" >&2
        exit 1 ;;
esac

n_scenarios=$(uv run python -c "import pandas,sys; print(len(pandas.read_csv(sys.argv[1])))" "$DATASET")

if [ "$group" = "--dry-run" ]; then
    echo "Dataset: $DATASET (${n_scenarios} scenarios), k=$K"
    echo
    printf '  %-8s %-8s %s\n' GROUP COST MODELS
    printf '  %-8s %-8s %s\n' probe  '~$0'   "${#batch1_models[@]} models x 1 scenario (id check)"
    printf '  %-8s %-8s %s\n' batch1 '~$62'  "${#batch1_models[@]} non-Claude (2 retired, unavailable)"
    printf '  %-8s %-8s %s\n' batch2 '~$130' "${#batch2_models[@]} Claude (Sonnet ~\$47, Opus ~\$82)"
    echo
    echo "  Everything routes through OpenRouter (\$100/month cap). The full 15-model"
    echo "  run is ~\$212, so it splits: batch1 this month, batch2 after the reset."
    echo "  Check actual spend at https://openrouter.ai/activity between batches."
    exit 0
fi

# This experiment writes only into its own directory. Refuse to touch any
# existing experiment's logs, and refuse to share a directory with logs that
# were not produced by this script.
case "$(basename "$LOG_DIR")" in
    abstract|baseline|broken|GDrive|Legacy|Mech-Design|reasoning|rq2-narrative|logs)
        echo "Refusing to write into '$LOG_DIR' — that is an existing experiment." >&2
        echo "Set LOG_DIR to a dedicated directory (default: logs/variance)." >&2
        exit 1 ;;
esac
if [ -d "$LOG_DIR" ] && compgen -G "$LOG_DIR/*.eval" > /dev/null; then
    foreign=$(find "$LOG_DIR" -name '*.eval' ! -name "*variance-k*" | head -1)
    if [ -n "$foreign" ]; then
        echo "Refusing to run: $LOG_DIR already holds logs from another run" >&2
        echo "  e.g. $foreign" >&2
        echo "Set LOG_DIR to an empty or variance-only directory." >&2
        exit 1
    fi
fi

mkdir -p "$LOG_DIR"   # --log-dir requires an existing directory

for model in "${models[@]}"; do
    model_clean=${model##*/}
    if compgen -G "$LOG_DIR/*variance-k${K}-${model_clean}*.eval" > /dev/null; then
        echo "==> $model_clean — already has a log, skipping"
        continue
    fi
    echo "==> $model_clean"
    uv run python3 -m eval.eval \
        --model-name "$model" \
        --dataset "$DATASET" \
        --times "$K" \
        --temperature "$TEMPERATURE" \
        --experiment-name "variance-k${K}-${model_clean}" \
        --log-dir "$LOG_DIR" \
        ${LIMIT:+--limit "$LIMIT"} \
      || { echo "!! $model_clean FAILED — continuing; re-run this group to retry"; continue; }
done

echo
echo "Group '$group' done. Logs in $LOG_DIR/"
ls "$LOG_DIR"/*.eval 2>/dev/null | wc -l | xargs echo "  logs present:"
