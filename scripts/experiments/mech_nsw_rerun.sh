#!/usr/bin/env bash
# Camera-ready RQ4 rerun: mechanism prompts with the corrected (own-min NSW) recommendation.
# Only the 342 scenarios whose prompt changes are rerun (data/mech_variants_nsw_changed/);
# the other 1,193 keep their original results. GPT-5 Mini additionally gets the 16
# non-contract variants on all 1,535 (the original folder held gpt-4o-mini runs there).
# Usage: scripts/experiments/mech_nsw_rerun.sh <openrouter model id> <folder name> [extra eval flags]
set -euo pipefail
MODEL=$1; NAME=$2; shift 2
OUT=logs/Mech-Design-nsw/$NAME; mkdir -p "$OUT"
for f in data/mech_variants_nsw_changed/*.csv; do
  v=$(basename "$f" .csv)
  data=$f
  if [[ $NAME == gpt-5-mini && $v != contracts-* ]]; then data=data/mech_variants_nsw/$v.csv; fi
  # skip only variants with a successfully completed log (an interrupted run leaves a "started" log)
  if uv run python - "$OUT" "$v" <<'PY' 2>/dev/null; then echo "skip $v"; continue; fi
import glob, sys
from inspect_ai.log import read_eval_log
ok = any(read_eval_log(f, header_only=True).status == "success"
         for f in glob.glob(f"{sys.argv[1]}/*_mech-nsw-{sys.argv[2]}_*.eval"))
sys.exit(0 if ok else 1)
PY
  uv run python3 -m eval.eval --dataset "$data" --model-name "$MODEL" --times 1 \
    --experiment-name "mech-nsw-$v" --log-dir "$OUT" "$@" > "$OUT/$v.out" 2>&1 && echo "done $v" || echo "FAILED $v"
done
