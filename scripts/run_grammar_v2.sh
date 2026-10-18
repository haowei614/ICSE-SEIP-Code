#!/usr/bin/env bash
# Grammar cells, one at a time. smoke = 3 items; full = n=40, resumable.
#   DATA_ROOT=/aisin-repair/apsec-v2-data bash scripts/run_grammar_v2.sh smoke
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
OUT=results/pillar1_grammar_v2
LOG="$OUT/run.log"
mkdir -p "$OUT"
MODE="${1:-full}"
MODELS=(qwen25_7b_instruct mistral_7b_instruct llama31_8b_instruct gemma2_9b_it)
SPECS=(TF_Mid:F16 TF_Reason:F16 TF_Mid:Q4_K_M TF_Reason:Q4_K_M)

run_cell() {
  local task="$1" model="$2" prec="$3" limit="${4:-}"
  echo "[cell] $(date -Is) $task $model $prec limit=${limit:-all}" | tee -a "$LOG"
  local -a extra=()
  if [[ -n "$limit" ]]; then
    extra+=(--limit "$limit")
  fi
  "$PY" -m src.run_experiments \
    --config config/pillar1_grammar_v2.yaml \
    --backend llama_cpp \
    --data-root "${DATA_ROOT:-data}" \
    --out-dir "$OUT" \
    --task "$task" --model "$model" --precision "$prec" \
    --resume \
    "${extra[@]}" >> "$LOG" 2>&1 || echo "[cell-error] $(date -Is) $task $model $prec" | tee -a "$LOG"
  return 0
}

echo "[start] $(date -Is) mode=$MODE py=$PY" | tee -a "$LOG"
limit=""
if [[ "$MODE" == "smoke" ]]; then
  limit=3
elif [[ "$MODE" != "full" ]]; then
  echo "usage: $0 smoke|full" >&2
  exit 2
fi
for spec in "${SPECS[@]}"; do
  task="${spec%%:*}"
  prec="${spec##*:}"
  for model in "${MODELS[@]}"; do
    run_cell "$task" "$model" "$prec" "$limit"
  done
done
echo "[done] $(date -Is) mode=$MODE" | tee -a "$LOG"
