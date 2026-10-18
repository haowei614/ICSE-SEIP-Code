#!/usr/bin/env bash
# Convert locked HF instruct checkpoints to llama.cpp F16 GGUF (same-stack control).
# Prefer this over third-party F16 GGUFs so weights match the transformers FP16 reference.
#
# Prerequisites on the 4090 server:
#   - HF weights under /workspace/models/<Name>
#   - llama.cpp checkout with convert_hf_to_gguf.py
#   - enough disk (~15–20GB per 7–9B F16 GGUF)
#
# Usage:
#   LLAMA_CPP_DIR=/path/to/llama.cpp ./scripts/convert_hf_to_gguf_f16.sh qwen25_7b_instruct
#   LLAMA_CPP_DIR=/path/to/llama.cpp ./scripts/convert_hf_to_gguf_f16.sh all
set -euo pipefail

ROOT="${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a llama.cpp checkout}"
OUT_ROOT="${OUT_ROOT:-/workspace/models/gguf}"
CONVERT="$ROOT/convert_hf_to_gguf.py"
if [[ ! -f "$CONVERT" ]]; then
  echo "missing $CONVERT" >&2
  exit 1
fi

declare -A HF_DIRS=(
  [qwen25_7b_instruct]=/workspace/models/Qwen2.5-7B-Instruct
  [mistral_7b_instruct]=/workspace/models/Mistral-7B-Instruct-v0.3
  [llama31_8b_instruct]=/workspace/models/Llama-3.1-8B-Instruct
  [gemma2_9b_it]=/workspace/models/Gemma-2-9B-it
)

models=("$@")
if [[ ${#models[@]} -eq 0 || "${models[0]}" == "all" ]]; then
  models=(qwen25_7b_instruct mistral_7b_instruct llama31_8b_instruct gemma2_9b_it)
fi

for mid in "${models[@]}"; do
  src="${HF_DIRS[$mid]:-}"
  if [[ -z "$src" || ! -d "$src" ]]; then
    echo "[skip] $mid: HF dir missing ($src)" >&2
    continue
  fi
  dest="$OUT_ROOT/$mid"
  mkdir -p "$dest"
  echo "[convert] $mid <- $src  (outtype f16)"
  python "$CONVERT" "$src" --outfile "$dest/${mid}-f16.gguf" --outtype f16
  echo "[ok] $dest/${mid}-f16.gguf"
done
