# Reviewer Package Manifest

## Included

- Experiment harness, parsers, metrics, provenance capture, and tests.
- Frozen local, cloud-reference, alignment, and grammar configurations.
- Locked public benchmark inputs and source notes.
- Primary FP16 and aligned GGUF per-item records.
- Legacy raw-prompt and superseded string-input records used for serving-stack
  comparisons.
- Stored cloud-reference outputs.
- Analysis scripts and generated CSV summaries.
- Failure coding, double-coding, visibility, and de-identified human-review
  records.
- De-identified practitioner materials and decision-table change logs.

## Deliberately excluded

- Manuscript sources and PDF.
- Model weights and local model caches.
- API credentials and environment files.
- Proprietary requirements or company documents.
- Direct participant identifiers.
- Early pilot, smoke-test, and superseded result directories that do not support
  a reported comparison.
- Internal manuscript planning and revision notes.

## Known availability boundary

The local checkout used to prepare this package does not contain the complete
`results/pillar1_grammar_v2/` per-item directory. The frozen grammar
configuration and execution/analysis code are included, but the complete
grammar run records must be added from the execution host before the package
can independently audit every grammar-decoding number.

The GGUF `peak_memory_mb` field is process maximum RSS as recorded by
`llama.cpp`, not a direct NVIDIA VRAM measurement. FP16 Transformers records
use `torch.cuda.max_memory_allocated`.
