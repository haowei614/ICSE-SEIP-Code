#!/usr/bin/env python3
"""Compare HF chat-template token ids with llama.cpp's completion prompt.

llama.cpp prepends a BOS token when the GGUF metadata says so. HF chat
templates for Mistral, Llama 3, and Gemma already include that token, so the
completion prompt starts with BOS twice.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_cpp import Llama

from src.prompt_format import apply_hf_chat_template, load_hf_tokenizer

PROMPT = (
    "Classify the software requirement into exactly one label.\n\n"
    "Requirement:\nThe system shall respond within one second.\n\nLabel:"
)
DECODING = {"use_chat_template": True, "enable_thinking": False}

MODELS = [
    (
        "qwen25_7b_instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "/workspace/models/gguf/qwen25_7b_instruct/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    ),
    (
        "mistral_7b_instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "/workspace/models/gguf/mistral_7b_instruct/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    ),
    (
        "llama31_8b_instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "/workspace/models/gguf/llama31_8b_instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    (
        "gemma2_9b_it",
        "google/gemma-2-9b-it",
        "/workspace/models/gguf/gemma2_9b_it/gemma-2-9b-it-Q4_K_M.gguf",
    ),
]


def main() -> int:
    for mid, hf, gguf in MODELS:
        print(f"\n=== {mid} ===", flush=True)
        tok = load_hf_tokenizer(hf)
        rendered = apply_hf_chat_template(tok, PROMPT, DECODING)
        hf_ids = tok(rendered, add_special_tokens=False)["input_ids"]
        llm = Llama(
            model_path=gguf, vocab_only=True, n_ctx=512, verbose=False,
        )
        bos = llm.token_bos()
        bos_piece = llm._model.token_get_text(bos)
        add_bos = bool(llm._model.add_bos_token())
        special = llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)
        as_called = ([bos] + special) if add_bos and bos != -1 else list(special)
        fixed = list(special)
        if add_bos and bos != -1 and (not fixed or fixed[0] != bos):
            fixed = [bos] + fixed
        print("rendered_prefix", repr(rendered[:60]))
        print("add_bos", add_bos, "bos_id", bos, "bos_piece", repr(bos_piece))
        print("hf", hf_ids[:8], "n", len(hf_ids))
        print("llama_as_called", as_called[:8], "n", len(as_called), "dup", as_called[:2] == [bos, bos])
        print("llama_fixed", fixed[:8], "n", len(fixed), "match_hf", fixed == hf_ids)
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
