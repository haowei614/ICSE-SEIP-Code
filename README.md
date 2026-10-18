# Replication Package: Local LLMs for Requirements Engineering

This repository contains the code, frozen configurations, public benchmark data,
per-item outputs, analysis scripts, failure-coding materials, and de-identified
human-review/practitioner artifacts used in the accompanying empirical study.
The manuscript itself is intentionally not included.

## Repository map

```text
config/       Frozen model, task, serving, grammar, and hardware configurations
data/         Locked public benchmark inputs and source notes
src/          Experiment harness, model backends, parsers, metrics, and provenance
scripts/      Download, conversion, parity-diagnostic, and execution helpers
results/      Primary and diagnostic per-item JSONL run records
analysis/     Aggregation, bootstrap, parser-sensitivity, and decision scripts
coding/       Failure codebook, samples, double coding, and human-review records
decision/     Decision-table inputs and de-identified practitioner feedback
tests/        Harness, parser, provenance, and metric tests
```

## Primary result sets

- `results/pillar1_fp16/`: Hugging Face Transformers FP16 reference.
- `results/pillar1_aligned_v2/`: primary GGUF F16/Q8/Q6/Q4 ladder with
  Hugging Face chat-template parity, token-id input, and a 1024-token cap.
- `results/pillar1_quant/` and `results/pillar1_f16_gguf/`: legacy raw-prompt
  runs retained as serving-misconfiguration evidence, not precision evidence.
- `results/pillar1_f16_gguf_aligned/` and
  `results/pillar1_quant_aligned/`: superseded string-input aligned runs with
  the documented BOS/cap confound.
- `results/pillar1_cloud/`: stored cloud upper-reference outputs.

Smoke runs, early pilots, and superseded diagnostics are omitted from the public
reviewer package unless they are needed to reproduce a reported contrast.

## Environment

The primary local envelope is an NVIDIA RTX 4090 (24 GB). FP16 runs use
Hugging Face Transformers; GGUF runs use `llama-cpp-python==0.3.8`.
Install the Python dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights are not redistributed. Update the local `hf` and
`gguf_by_precision` paths in the selected YAML configuration, or reproduce the
directory layout documented there. `gguf_manifest.csv` records the expected
GGUF files and hashes.

## Validate the package

From the repository root:

```bash
python -m tests.test_pipeline
python -m tests.test_harness
python -m tests.test_output_parsers
```

The harness also supports a mock-only integrity check:

```bash
python -m src.run_experiments --config config/pilot.yaml --dry-run
```

Mock records are tagged with `"mock": true`; aggregation code rejects them as
empirical evidence.

## Reproduce the primary analyses

```bash
python -m analysis.aggregate_aligned_v2 --out analysis/out
python -m analysis.paper_fix_pass
python -m analysis.parser_sensitivity
python -m analysis.human_review coding/human_review_sheet.csv \
  --out analysis/out
```

Analysis outputs are deterministic under the seeds recorded in the scripts and
configs. `analysis/README.md` documents the bootstrap procedure.

## Run the aligned GGUF ladder

After downloading the models and updating local paths:

```bash
bash scripts/run_aligned_v2.sh
```

The frozen primary configuration is `config/pillar1_aligned_v2.yaml`. The run
records include prompt hashes, model/precision identity, seeds, latency, token
counts, malformed/OOM/timeout flags, and a provenance block. Local absolute
paths in configs document the execution host layout and may be replaced.

## Data and metrics

The study uses one labeled measurement basis:

- TF-Surface: PROMISE NFR subtype classification, item accuracy.
- TF-Mid: DAMIR-PURE anaphoric-span detection, the locked P3-A score.
- TF-Reason: eTour requirement-to-code trace recovery, item link F1.

Source and transformation notes are in `data/SOURCES.md`. Dataset material
retains its upstream terms; inclusion here does not relicense third-party data.

## Failure and human-review evidence

`coding/` contains the versioned codebook, sampled packets, primary and
second-coder sheets, visibility labels, and de-identified review records.
Wave 1 is FP16. The coded Wave 2 and 15 cases in the mixed human-review
sensitivity sample come from the legacy raw-prompt quantized stack; the package
does not present them as post-alignment prevalence estimates.

`decision/` contains the decision-table construction and de-identified formative
feedback. Participant identifiers are study IDs only.

## Integrity and privacy

- No model weights, API keys, proprietary requirements, company documents, or
  direct participant identifiers are included.
- Cloud credentials are read only from environment-variable names in configs.
- Public benchmark exhibits are the only textual cases released.
- Failed runs remain in JSONL records and are not silently dropped.

## License

Code and original documentation are released under the MIT License. Public
datasets and model outputs remain subject to their upstream terms.
