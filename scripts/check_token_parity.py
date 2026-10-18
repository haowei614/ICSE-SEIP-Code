#!/usr/bin/env python3
"""Compare token ids for the FP16 path, the old llama.cpp string path, and the new id path.

Uses the first 3 items of the n=40 eval sample (sampling seed 20260905).
Does not generate. Loads each model's Q4 GGUF with vocab_only=True.

Exit 1 if Path1 != Path3, or if Path1 != the ids the transformers backend
actually builds (chat text, then tokenizer call with add_special_tokens=True
and left truncation).
"""
from __future__ import annotations

import inspect
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import tasks as tasksmod
from src.prompt_format import apply_hf_chat_template, load_hf_tokenizer
from src.sampling import stratified_sample


def _as_id_list(ids) -> list[int]:
    """apply_chat_template(tokenize=True) may return a list, a tensor, or a BatchEncoding."""
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def path1_ids(tok, prompt: str, decoding: dict) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=True, add_generation_prompt=True)
    try:
        ids = tok.apply_chat_template(
            messages, enable_thinking=bool(decoding.get("enable_thinking", False)), **kwargs
        )
    except TypeError:
        ids = tok.apply_chat_template(messages, **kwargs)
    return _as_id_list(ids)


def live_fp16_ids(tok, prompt: str, decoding: dict, max_input: int) -> list[int]:
    text = apply_hf_chat_template(tok, prompt, decoding)
    if getattr(tok, "truncation_side", None) is not None:
        tok.truncation_side = "left"
    encoded = tok(text, truncation=True, max_length=max_input, add_special_tokens=True)
    return [int(x) for x in encoded["input_ids"]]


def path3_ids(tok, prompt: str, decoding: dict) -> list[int]:
    text = apply_hf_chat_template(tok, prompt, decoding)
    return [int(x) for x in tok(text, add_special_tokens=False)["input_ids"]]


def string_path_ids(llm, text: str) -> list[int]:
    """Ids llama-cpp-python 0.3.8 ``_create_completion`` feeds for a string prompt.

    ``tokenize`` is called with ``add_bos=False`` and ``special=True``. A leading
    BOS/CLS is prepended only when the GGUF metadata says ``add_bos_token``, and
    an EOS/SEP is appended when ``add_eos_token`` is set. This is the string
    path used by ``self._llm(text_in)``.
    """
    bos_token_id = llm.token_bos()
    cls_token_id = llm._model.token_cls()
    sep_token_id = llm._model.token_sep()
    bos_tokens = [cls_token_id if cls_token_id != -1 else bos_token_id]
    eos_tokens = [sep_token_id if sep_token_id != -1 else llm.token_eos()]
    if (not llm._model.add_bos_token()) or bos_tokens[:1] == [-1]:
        bos_tokens = []
    if (not llm._model.add_eos_token()) and sep_token_id == -1:
        eos_tokens = []
    prefix = (
        list(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
        if text != "" else []
    )
    return bos_tokens + prefix + eos_tokens


def first_diff(a: list[int], b: list[int], tok) -> str:
    if a == b:
        return ""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return (
                f"pos={i} len={len(a)}/{len(b)} "
                f"P_left={a[i]}:{tok.convert_ids_to_tokens(a[i])!r} "
                f"P_right={b[i]}:{tok.convert_ids_to_tokens(b[i])!r}"
            )
    return f"prefix equal; len={len(a)}/{len(b)}"


def sample_items(cfg, task_cfg, data_root: str, n_show: int):
    items = tasksmod.load_task(task_cfg, data_root=data_root)
    samp = cfg.get("sampling") or {}
    items = stratified_sample(
        items,
        int(samp.get("size_per_task") or len(items)),
        seed=int(samp.get("seed", cfg.get("seed", 0))),
        id_of=lambda it: it.item_id,
        stratum_of=lambda it: (
            str((it.reference or {}).get("gold_label"))
            if isinstance(it.reference, dict) and (it.reference or {}).get("gold_label") is not None
            else None
        ),
    )
    return items[:n_show]


def main() -> int:
    from llama_cpp import Llama

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    with open("config/pillar1_aligned_v2.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    decoding = dict(cfg.get("decoding") or {})
    max_input = int(decoding.get("max_input_tokens") or 8192)
    data_root = "data"
    rows = []
    failed = False
    for model in cfg["models"]:
        hf_name = model.get("hf") or model.get("hf_id")
        gguf = (model.get("gguf_by_precision") or {})["Q4_K_M"]
        print(f"[load] {model['id']} tokenizer={hf_name}", flush=True)
        tok = load_hf_tokenizer(hf_name, model.get("revision"))
        print(f"[load] vocab_only {gguf}", flush=True)
        llm = Llama(model_path=gguf, vocab_only=True, n_ctx=512, verbose=False)
        try:
            add_bos_default = inspect.signature(llm.tokenize).parameters["add_bos"].default
        except Exception:
            add_bos_default = "unknown"
        print(
            f"[tokenize] tokenize.add_bos default={add_bos_default!r} "
            f"gguf.add_bos_token={llm._model.add_bos_token()} "
            f"gguf.add_eos_token={llm._model.add_eos_token()} "
            f"bos={llm.token_bos()} eos={llm.token_eos()}",
            flush=True,
        )
        for task_cfg in cfg["tasks"]:
            items = sample_items(cfg, task_cfg, data_root, 3)
            for item in items:
                prompt = tasksmod.build_prompt(task_cfg, item)
                text = apply_hf_chat_template(tok, prompt, decoding)
                p1 = path1_ids(tok, prompt, decoding)
                p2 = string_path_ids(llm, text)
                p3 = path3_ids(tok, prompt, decoding)
                live = live_fp16_ids(tok, prompt, decoding, max_input)
                p1_p2 = p1 == p2
                p1_p3 = p1 == p3
                p1_live = p1 == live
                diff = ""
                if not p1_p3:
                    diff = "P1!=P3 " + first_diff(p1, p3, tok)
                    failed = True
                elif not p1_p2:
                    diff = "P1!=P2 " + first_diff(p1, p2, tok)
                if not p1_live:
                    diff = (diff + " | " if diff else "") + "P1!=liveFP16 " + first_diff(p1, live, tok)
                    failed = True
                rows.append((model["id"], task_cfg["id"], item.item_id, p1_p2, p1_p3, p1_live, diff))
                print(
                    f"{model['id']} {task_cfg['id']} {item.item_id} "
                    f"P1==P2={p1_p2} P1==P3={p1_p3} P1==live={p1_live} {diff}",
                    flush=True,
                )
        llm.close()
    print("\nmodel | task | item | P1==P2 | P1==P3 | P1==liveFP16 | first difference")
    for model_id, task, item_id, p1_p2, p1_p3, p1_live, diff in rows:
        print(f"{model_id} | {task} | {item_id} | {p1_p2} | {p1_p3} | {p1_live} | {diff}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
