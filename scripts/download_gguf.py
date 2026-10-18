#!/usr/bin/env python3
"""Download bartowski GGUF files for the locked Pillar-1 quant ladder.

Default: Q4_K_M + Q8_0 (+ optional Q6_K) into /workspace/models/gguf/<model_id>/
Uses HF token from /aisin-repair/.hf_token when present.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# (local_id, hf_gguf_repo, filename patterns by precision)
# Prefer scripts/convert_hf_to_gguf_f16.sh for F16 (same weights as transformers FP16).
# Bartowski publishes Qwen f16; Llama/Mistral bartowski repos ship f32 (too large for
# 24GB without offload); Gemma bartowski has f32 / Q8-f16 hybrids, not clean F16.
MODELS = {
    "qwen25_7b_instruct": (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        {
            "F16": "Qwen2.5-7B-Instruct-f16.gguf",
            "Q4_K_M": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "Q6_K": "Qwen2.5-7B-Instruct-Q6_K.gguf",
            "Q8_0": "Qwen2.5-7B-Instruct-Q8_0.gguf",
        },
    ),
    "mistral_7b_instruct": (
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        {
            "Q4_K_M": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            "Q6_K": "Mistral-7B-Instruct-v0.3-Q6_K.gguf",
            "Q8_0": "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
        },
    ),
    "llama31_8b_instruct": (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        {
            "Q4_K_M": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "Q6_K": "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
            "Q8_0": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        },
    ),
    "gemma2_9b_it": (
        "bartowski/gemma-2-9b-it-GGUF",
        {
            "Q4_K_M": "gemma-2-9b-it-Q4_K_M.gguf",
            "Q6_K": "gemma-2-9b-it-Q6_K.gguf",
            "Q8_0": "gemma-2-9b-it-Q8_0.gguf",
        },
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS) + ["all"], default="qwen25_7b_instruct")
    ap.add_argument("--precisions", default="Q4_K_M,Q8_0",
                    help="Comma list from F16,Q4_K_M,Q6_K,Q8_0")
    ap.add_argument("--out-root", default="/workspace/models/gguf")
    args = ap.parse_args()

    token = None
    for p in ("/aisin-repair/.hf_token", os.path.expanduser("~/.cache/huggingface/token")):
        if os.path.isfile(p):
            token = Path(p).read_text(encoding="utf-8").strip()
            break
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    from huggingface_hub import hf_hub_download

    precisions = [x.strip() for x in args.precisions.split(",") if x.strip()]
    models = list(MODELS) if args.model == "all" else [args.model]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for mid in models:
        repo, files = MODELS[mid]
        dest = out_root / mid
        dest.mkdir(parents=True, exist_ok=True)
        for prec in precisions:
            if prec not in files:
                raise SystemExit(f"unknown precision {prec} for {mid}")
            fname = files[prec]
            print(f"[download] {repo} / {fname}", flush=True)
            path = hf_hub_download(
                repo_id=repo,
                filename=fname,
                local_dir=str(dest),
                token=token,
            )
            print(f"[ok] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
