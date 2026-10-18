# Pillar 2 Wave 2 — quant failures

**n=52**, seed=20260907. Source: `results/pillar1_quant` only (Q8/Q6/Q4).  
Together with Wave 1 (48 FP16 cases) ≈ **100** toward the 100–150 target.

## Strata
- task: {'TF_Mid': 12, 'TF_Surface': 12, 'TF_Reason': 28}
- model: {'llama31_8b_instruct': 12, 'gemma2_9b_it': 13, 'mistral_7b_instruct': 14, 'qwen25_7b_instruct': 13}
- quant: {'Q4_K_M': 16, 'Q6_K': 17, 'Q8_0': 19}
- fail_kind: {'malformed': 27, 'zero': 18, 'partial': 7}

## How to code
Same as Wave 1: fill `wave2_coding_sheet.csv` using `wave2_packets.jsonl` + `../codebook.md`.  
See `HOW_TO_CODE_zh.md`. Prefer checking whether failures are **M5** (format under llama.cpp) vs **M6** (well-formed wrong).
