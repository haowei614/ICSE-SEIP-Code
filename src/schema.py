"""Run-record schema. One record per (task, dataset, model, precision, item, rep).

Design goals (PART 12):
  - Capture EVERYTHING needed for Pillar 1 (quality/resource) and Pillar 2
    (failure interpretation) so no rerun is needed to inspect a failure.
  - NEVER discard a failed/malformed/OOM/timeout run: those are recorded with the
    relevant flag set, not dropped.
  - `mock=True` records can never populate the paper (analysis guards enforce it).
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


SCHEMA_VERSION = "2.0"


@dataclass
class RunRecord:
    # ---- experiment / cell identity ----
    experiment_id: str                 # groups all runs of one invocation
    task_family: str                   # TF_Surface | TF_Mid | TF_Reason
    task_name: str                     # e.g. nfr_classification
    dataset: str
    item_id: str
    model_id: str
    precision: str                     # FP16/BF16/Q8_0/Q6_K/Q4_K_M ...
    measurement_basis: str             # "A" or "B"
    seed: int
    rep: int = 0                       # repetition index (>=0)

    # ---- provenance ----
    model_revision: Optional[str] = None
    prompt_version: Optional[str] = None
    prompt_sha256: Optional[str] = None
    backend: str = "unknown"
    mock: bool = False
    schema_version: str = SCHEMA_VERSION
    timestamp: float = field(default_factory=time.time)
    run_uuid: str = field(default_factory=lambda: uuid.uuid4().hex)

    # ---- model interaction ----
    input_length_chars: Optional[int] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw_model_output: Optional[str] = None
    parsed_output: Any = None
    gold_label: Any = None

    # ---- quality (basis A: labeled metric contribution) ----
    quality: Optional[float] = None
    quality_detail: dict[str, Any] = field(default_factory=dict)

    # ---- resources ----
    total_latency_s: Optional[float] = None
    ttft_s: Optional[float] = None
    tokens_per_s: Optional[float] = None
    peak_memory_mb: Optional[float] = None      # unified mem on Apple; VRAM on CUDA
    peak_memory_method: Optional[str] = None

    # ---- integrity / failure flags (never silently dropped) ----
    malformed_output: bool = False
    oom: bool = False
    timeout: bool = False
    exception: Optional[str] = None
    finish_reason: Optional[str] = None
    # git commit, package/CUDA versions, resolved decoding, GGUF hash.
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_records(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
