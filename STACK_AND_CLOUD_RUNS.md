# Stack-control + cloud upper-bound runs

## 1. Cloud frontier (Claude Sonnet 4.5)

Already wired. From `replication/`:

```bash
export ANTHROPIC_API_KEY=...   # present in author env
python -m src.run_experiments \
  --config config/pillar1_cloud_ref.yaml \
  --backend cloud \
  --data-root data \
  --out-dir results/pillar1_cloud
```

OpenAI alternative: set `provider: openai`, `api_model: gpt-4.1` (or locked id),
`key_env: OPENAI_API_KEY` in the same YAML.

After completion:

```bash
python analysis/aggregate_pillar1.py results/pillar1_cloud --out analysis/out_cloud
# then fill macros Qual*Cloud / Delta* from the CSV (never invent)
```

## 2. F16 GGUF same-stack control (RTX 4090)

Convert the **same HF trees** used for transformers FP16 (preferred; weight identity):

```bash
cd /aisin-repair/APSEC-My/replication
LLAMA_CPP_DIR=/path/to/llama.cpp ./scripts/convert_hf_to_gguf_f16.sh all
# Gemma-2-9B F16 is tight on 24GB; if OOM, drop that model from the YAML.
python -m src.run_experiments \
  --config config/pillar1_f16_gguf.yaml \
  --backend llama_cpp \
  --data-root data \
  --out-dir results/pillar1_f16_gguf
```

Qwen-only shortcut (bartowski already ships `*-f16.gguf`):

```bash
python scripts/download_gguf.py --model qwen25_7b_instruct --precisions F16
# then point gguf_by_precision.F16 at the bartowski filename if not using convert
```

Do **not** use bartowski F32 GGUFs on 24GB (will not fit), and do not use
`Q8_0-f16` hybrids as a full-precision control.

Reading rule:
- **F16 → Q\*** (same llama.cpp): bit-width / K-quant effect
- **transformers FP16 → F16 GGUF**: serving-stack effect at matched precision

## 4. Chat-template alignment (critical confound check)

Historical GGUF/F16 runs used **raw task prompts** (`use_chat_template: false`)
while transformers FP16 used HF `apply_chat_template`. That alone can inflate
malformed rates under llama.cpp.

Aligned path (after harness fix):

```bash
# 1) Dump rendered prompts (no generation)
.venv/bin/python scripts/compare_prompt_tokens.py \
  --config config/pillar1_f16_gguf_aligned.yaml \
  --model-id qwen25_7b_instruct --task TF_Reason --limit 5 --legacy-raw \
  --out analysis/out/prompt_token_diff_qwen_reason.jsonl

# 2) Re-run F16 with chat-template parity
.venv/bin/python -m src.run_experiments \
  --config config/pillar1_f16_gguf_aligned.yaml \
  --backend llama_cpp --data-root data \
  --out-dir results/pillar1_f16_gguf_aligned

# 3) Optional: same + JSON GBNF grammar (Mid/Reason)
.venv/bin/python -m src.run_experiments \
  --config config/pillar1_f16_gguf_grammar.yaml \
  --backend llama_cpp --data-root data \
  --out-dir results/pillar1_f16_gguf_grammar
```

Reading rule after alignment:
- If Mid/Reason mal drops sharply vs legacy F16 → prior "stack effect" was partly
  a **harness chat-template bug**; revise claims toward serving-config sensitivity.
- If mal stays high after alignment → harder deployment-stack finding.
- If grammar drives mal→0 but semantic F1 stays low → DO_NOT without grammar is
  too strong; table should be path × grammar.
