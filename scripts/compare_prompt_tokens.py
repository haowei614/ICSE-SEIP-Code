#!/usr/bin/env python3
"""Compare the *actual* prompt text / token ids fed to transformers vs llama.cpp.

Purpose: isolate whether the FP16→F16 malformed jump is a serving-stack effect
or a chat-template / raw-completion harness mismatch.

Does NOT run generation. Loads HF tokenizer (+ optional llama.cpp tokenizer) only.

Example (on the 4090 node)::

  cd /aisin-repair/APSEC-My/replication
  .venv/bin/python scripts/compare_prompt_tokens.py \\
    --config config/pillar1_f16_gguf_aligned.yaml \\
    --model-id qwen25_7b_instruct \\
    --task TF_Reason --limit 3 \\
    --out analysis/out/prompt_token_diff.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import yaml

# Allow `python scripts/...` from replication/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import tasks as tasksmod
from src.prompt_format import apply_hf_chat_template, load_hf_tokenizer
from src.sampling import stratified_sample


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--task", default="TF_Reason")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", default="analysis/out/prompt_token_diff.jsonl")
    ap.add_argument(
        "--legacy-raw",
        action="store_true",
        help="Also dump the OLD llama.cpp path (raw task prompt, no chat template).",
    )
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    model = next(m for m in cfg["models"] if m["id"] == args.model_id)
    task_cfg = next(t for t in cfg["tasks"] if t["id"] == args.task)
    decoding = dict(cfg.get("decoding", {}))

    hf_name = model.get("hf") or model.get("hf_id")
    if not hf_name:
        raise SystemExit("model needs hf or hf_id for chat-template parity")

    print(f"[load] HF tokenizer {hf_name}", flush=True)
    tok = load_hf_tokenizer(hf_name, model.get("revision"))

    items = tasksmod.load_task(task_cfg, data_root=args.data_root)
    samp = cfg.get("sampling") or {}
    if samp.get("size_per_task"):
        items = stratified_sample(
            items,
            int(samp["size_per_task"]),
            seed=int(samp.get("seed", cfg.get("seed", 0))),
            id_of=lambda it: it.item_id,
            stratum_of=lambda it: (
                (it.reference or {}).get("gold_label")
                or (it.reference or {}).get("stratum")
                or "all"
            ),
        )
    items = items[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_mismatch = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for item in items:
            raw = tasksmod.build_prompt(task_cfg, item)
            rendered = apply_hf_chat_template(tok, raw, decoding)
            # Path A: transformers (HF tokenize of rendered chat text)
            ids_hf = tok(rendered, add_special_tokens=False)["input_ids"]
            # Path B (aligned): same rendered string; llama.cpp would tokenize it
            ids_legacy = tok(raw, add_special_tokens=False)["input_ids"] if args.legacy_raw else None

            same = rendered == raw
            row = {
                "item_id": item.item_id,
                "task_family": task_cfg["id"],
                "model_id": model["id"],
                "use_chat_template": bool(decoding.get("use_chat_template", True)),
                "raw_sha256": sha256(raw),
                "rendered_sha256": sha256(rendered),
                "raw_equals_rendered": same,
                "n_tokens_hf_rendered": len(ids_hf),
                "rendered_prefix": rendered[:240].replace("\n", "\\n"),
                "rendered_suffix": rendered[-240:].replace("\n", "\\n"),
            }
            if ids_legacy is not None:
                row["n_tokens_hf_raw"] = len(ids_legacy)
                row["raw_prefix"] = raw[:240].replace("\n", "\\n")
                # The historical bug: llama.cpp completed on raw while HF used rendered.
                row["legacy_mismatch"] = not same
                if not same:
                    n_mismatch += 1
            elif not same and decoding.get("use_chat_template", True):
                n_mismatch += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{item.item_id}] raw==rendered={same} "
                f"tok_rendered={len(ids_hf)} "
                f"sha_r={row['rendered_sha256'][:10]}",
                flush=True,
            )

    print(
        f"[done] wrote {args.out} items={len(items)} "
        f"chat_wrap_changes={n_mismatch}/{len(items)} "
        f"(non-zero means chat template alters the prompt)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
