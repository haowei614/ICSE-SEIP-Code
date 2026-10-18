#!/usr/bin/env python3
"""Wave 2 Pillar-2 sample: stratified failures from quant cells (Q8/Q6/Q4).

Complements Wave 1 (FP16-only). Target: grow coded N toward 100–150.
Failure = quality < 1.0. Prefer Reason + high-malformed strata.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sampling import stratified_sample  # noqa: E402
from analysis.sample_pillar2 import _fail_kind, _load_item_text  # noqa: E402

SEED = 20260907
REASON_N, SURFACE_N, MID_N = 28, 12, 12  # total 52 → with Wave1 48 ≈ 100


def load_quant(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("TF_*__*__Q*.jsonl")):
        parts = path.name.replace(".jsonl", "").split("__")
        task, model, prec = parts[0], parts[1], parts[2]
        for rec in map(json.loads, path.open(encoding="utf-8")):
            rec = dict(rec)
            rec["_task"] = task
            rec["_model"] = model
            rec["_precision"] = prec
            rec["_source_file"] = path.name
            if _fail_kind(rec) not in ("ok", "unscored"):
                rows.append(rec)
    return rows


def main() -> int:
    results = ROOT / "results" / "pillar1_quant"
    out_dir = ROOT / "coding" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    fails = load_quant(results)
    items = _load_item_text(ROOT / "data")

    by_task: dict[str, list] = defaultdict(list)
    for r in fails:
        by_task[r["_task"]].append(r)

    quotas = {"TF_Reason": REASON_N, "TF_Surface": SURFACE_N, "TF_Mid": MID_N}
    picked: list[dict] = []
    for task, want in quotas.items():
        pool = by_task.get(task, [])
        take = min(want, len(pool))
        sample = stratified_sample(
            pool, take, SEED,
            id_of=lambda r: f"{r['_model']}:{r['_precision']}:{r['item_id']}:{r.get('run_uuid','')}",
            stratum_of=lambda r: f"{r['_model']}|{r['_precision']}|{_fail_kind(r)}",
        )
        picked.extend(sample)
    picked = stratified_sample(
        picked, None, SEED,
        id_of=lambda r: f"{r['_task']}:{r['_model']}:{r['_precision']}:{r['item_id']}",
    )

    manifest = out_dir / "wave2_manifest.csv"
    sheet = out_dir / "wave2_coding_sheet.csv"
    packets = out_dir / "wave2_packets.jsonl"
    readme = out_dir / "WAVE2_README.md"

    with manifest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "case_id", "task_id", "model_id", "quant", "item_id", "fail_kind",
            "quality", "malformed", "experiment_id", "source_file", "run_uuid",
        ])
        w.writeheader()
        for i, r in enumerate(picked, 1):
            w.writerow({
                "case_id": f"W2-{i:03d}",
                "task_id": r["_task"],
                "model_id": r["_model"],
                "quant": r["_precision"],
                "item_id": r["item_id"],
                "fail_kind": _fail_kind(r),
                "quality": r.get("quality"),
                "malformed": int(bool(r.get("malformed_output"))),
                "experiment_id": r.get("experiment_id"),
                "source_file": r["_source_file"],
                "run_uuid": r.get("run_uuid"),
            })

    fields = [
        "case_id", "task_id", "model_id", "quant", "item_id",
        "primary_code", "secondary_codes", "visibility", "quality_attribute",
        "rationale", "quote_span", "coder",
    ]
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(picked, 1):
            w.writerow({
                "case_id": f"W2-{i:03d}",
                "task_id": r["_task"],
                "model_id": r["_model"],
                "quant": r["_precision"],
                "item_id": r["item_id"],
                "primary_code": "",
                "secondary_codes": "",
                "visibility": "",
                "quality_attribute": "",
                "rationale": "",
                "quote_span": "",
                "coder": "coderA",
            })

    with packets.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(picked, 1):
            meta = items.get((r["_task"], r["item_id"]), {})
            fh.write(json.dumps({
                "case_id": f"W2-{i:03d}",
                "task_id": r["_task"],
                "model_id": r["_model"],
                "quant": r["_precision"],
                "item_id": r["item_id"],
                "fail_kind": _fail_kind(r),
                "quality": r.get("quality"),
                "quality_detail": r.get("quality_detail"),
                "malformed_output": bool(r.get("malformed_output")),
                "input_text": meta.get("input_text", ""),
                "gold": meta.get("gold"),
                "gold_label": meta.get("gold_label"),
                "raw_model_output": r.get("raw_model_output") or "",
                "parsed_output": r.get("parsed_output"),
                "coding_hint": "Same codebook as Wave 1 (HOW_TO_CODE_zh.md). Quant cells often M5.",
            }, ensure_ascii=False) + "\n")

    tc, mc, pc, kc = (
        Counter(r["_task"] for r in picked),
        Counter(r["_model"] for r in picked),
        Counter(r["_precision"] for r in picked),
        Counter(_fail_kind(r) for r in picked),
    )
    readme.write_text(
        f"""# Pillar 2 Wave 2 — quant failures

**n={len(picked)}**, seed={SEED}. Source: `results/pillar1_quant` only (Q8/Q6/Q4).  
Together with Wave 1 (48 FP16 cases) ≈ **{48 + len(picked)}** toward the 100–150 target.

## Strata
- task: {dict(tc)}
- model: {dict(mc)}
- quant: {dict(pc)}
- fail_kind: {dict(kc)}

## How to code
Same as Wave 1: fill `wave2_coding_sheet.csv` using `wave2_packets.jsonl` + `../codebook.md`.  
See `HOW_TO_CODE_zh.md`. Prefer checking whether failures are **M5** (format under llama.cpp) vs **M6** (well-formed wrong).
""",
        encoding="utf-8",
    )
    print(f"[ok] wave2 n={len(picked)} tasks={dict(tc)} quants={dict(pc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
