#!/bin/bash
# Download HF weights, convert to F16 GGUF, delete the HF dir, then run
# retokenized Surface/Mid/Reason. Does not touch pillar1_f16_gguf_aligned.
set -euo pipefail
export HF_HOME=/workspace/hf-cache
export HF_TOKEN="$(cat /aisin-repair/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTHONUNBUFFERED=1
PY=/aisin-repair/APSEC-My/replication/.venv/bin/python
CONVERT=/workspace/llama.cpp/convert_hf_to_gguf.py
ROOT=/aisin-repair/APSEC-My/replication

download() {
  local repo="$1" dest="$2"
  echo "[download] $repo"
  "$PY" - "$repo" "$dest" << 'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    sys.argv[1],
    local_dir=sys.argv[2],
    ignore_patterns=["*.bin", "original/*", "consolidated*", "*.gguf", "*.ot", "flax_model*"],
)
print("[ok] download", sys.argv[2], flush=True)
PY
}

convert_one() {
  local mid="$1" src="$2"
  local dest="/workspace/models/gguf/${mid}/${mid}-f16.gguf"
  mkdir -p "/workspace/models/gguf/${mid}"
  echo "[convert] $mid"
  "$PY" "$CONVERT" "$src" --outfile "$dest" --outtype f16
  local bytes
  bytes=$(stat -c%s "$dest")
  if [ "$bytes" -lt 10000000000 ]; then
    echo "[fail] $dest is only $bytes bytes" >&2
    exit 1
  fi
  echo "[ok] $dest ($bytes)"
  rm -rf "$src"
  echo "[rm] $src"
}

if kill -0 11534 2>/dev/null; then
  echo "[wait] mistral download pid 11534"
  while kill -0 11534 2>/dev/null; do sleep 20; done
fi
if ! grep -q '\[ok\] mistral' /workspace/tmp/hf_download_mistral.log; then
  echo "[fail] mistral download did not finish cleanly" >&2
  tail -30 /workspace/tmp/hf_download_mistral.log >&2
  exit 1
fi
convert_one mistral_7b_instruct /workspace/models/Mistral-7B-Instruct-v0.3

download meta-llama/Llama-3.1-8B-Instruct /workspace/models/Llama-3.1-8B-Instruct
convert_one llama31_8b_instruct /workspace/models/Llama-3.1-8B-Instruct

download google/gemma-2-9b-it /workspace/models/Gemma-2-9B-it
convert_one gemma2_9b_it /workspace/models/Gemma-2-9B-it

cd "$ROOT"
echo "[run] retokenized F16"
"$PY" -m src.run_experiments \
  --config config/pillar1_f16_gguf_retok.yaml \
  --backend llama_cpp \
  --data-root data \
  --out-dir results/pillar1_f16_gguf_retok
echo "[done] retokenized F16"
