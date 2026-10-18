#!/usr/bin/env bash
# One aligned-v2 cell at a time. Resume is safe: completed item ids are skipped
# when the config hash and git commit match.
#
#   bash scripts/run_aligned_v2.sh smoke
#   bash scripts/run_aligned_v2.sh full
#   nohup bash scripts/run_aligned_v2.sh full > results/pillar1_aligned_v2/nohup.out 2>&1 &
#   tail -f results/pillar1_aligned_v2/run.log
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
OUT=results/pillar1_aligned_v2
LOG="$OUT/run.log"
mkdir -p "$OUT"
MODE="${1:-full}"

MODELS=(
  qwen25_7b_instruct
  mistral_7b_instruct
  llama31_8b_instruct
  gemma2_9b_it
)

run_cell() {
  local task="$1" model="$2" prec="$3" limit="${4:-}"
  echo "[cell] $(date -Is) $task $model $prec limit=${limit:-all}" | tee -a "$LOG"
  local -a extra=()
  if [[ -n "$limit" ]]; then
    extra+=(--limit "$limit")
  fi
    "$PY" -m src.run_experiments \
    --config config/pillar1_aligned_v2.yaml \
    --backend llama_cpp \
    --data-root "${DATA_ROOT:-data}" \
    --out-dir "$OUT" \
    --task "$task" --model "$model" --precision "$prec" \
    --resume \
    "${extra[@]}" >> "$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[cell-error] $(date -Is) $task $model $prec exit=$rc" | tee -a "$LOG"
  fi
  return 0
}

run_group() {
  local limit="$1"
  shift
  local spec
  for spec in "$@"; do
    local task="${spec%%:*}"
    local prec="${spec##*:}"
    local model
    for model in "${MODELS[@]}"; do
      run_cell "$task" "$model" "$prec" "$limit"
    done
  done
}

# Priority: Mid/Reason F16, Mid/Reason Q4, Surface F16 + Q4,
# Mid/Reason Q8, Mid/Reason Q6, Surface Q8/Q6.
FULL_SPECS=(
  TF_Mid:F16
  TF_Reason:F16
  TF_Mid:Q4_K_M
  TF_Reason:Q4_K_M
  TF_Surface:F16
  TF_Surface:Q4_K_M
  TF_Mid:Q8_0
  TF_Reason:Q8_0
  TF_Mid:Q6_K
  TF_Reason:Q6_K
  TF_Surface:Q8_0
  TF_Surface:Q6_K
)
SMOKE_SPECS=(
  TF_Surface:Q4_K_M
  TF_Mid:Q4_K_M
  TF_Reason:Q4_K_M
)

echo "[start] $(date -Is) mode=$MODE py=$PY" | tee -a "$LOG"
case "$MODE" in
  smoke)
    run_group 3 "${SMOKE_SPECS[@]}"
    ;;
  full)
    run_group "" "${FULL_SPECS[@]}"
    ;;
  *)
    echo "usage: $0 smoke|full" >&2
    exit 2
    ;;
esac
echo "[done] $(date -Is) mode=$MODE" | tee -a "$LOG"
