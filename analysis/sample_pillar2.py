#!/usr/bin/env python3
"""Draw a deterministic stratified Pillar-2 failure sample (Wave 1).

Failure = quality < 1.0 under measurement basis A (partial credit still codes).
Strata: task_family × model × fail_kind (malformed | zero | partial).

Outputs under coding/samples/:
  wave1_manifest.csv   — ids + strata + quality flags
  wave1_coding_sheet.csv — empty sheet matching coding_sheet_template.csv
  wave1_packets.jsonl  — one record per case with input/gold/output for coding
  WAVE1_README.md      — how to code this wave

Integrity: never invent codes; only selects existing logged failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sampling import stratified_sample  # noqa: E402

SEED = 20260907
DEFAULT_N = 48  # first wave; grow toward 100–150 after quants land


def _fail_kind(rec: dict) -> str:
    if rec.get("malformed_output"):
        return "malformed"
    q = rec.get("quality")
    if q is None:
        return "unscored"
    if q <= 0:
        return "zero"
    if q < 1.0:
        return "partial"
    return "ok"


def _load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("TF_*__*__FP16.jsonl")):
        parts = path.name.replace(".jsonl", "").split("__")
        task, model, precision = parts[0], parts[1], parts[2]
        for rec in map(json.loads, path.open(encoding="utf-8")):
            rec = dict(rec)
            rec["_task"] = task
            rec["_model"] = model
            rec["_precision"] = precision
            rec["_source_file"] = path.name
            rows.append(rec)
    return rows


def _load_item_text(data_root: Path) -> dict[tuple[str, str], dict]:
    """(task, item_id) -> {prompt_text, gold_summary}."""
    out: dict[tuple[str, str], dict] = {}
    # Surface
    p = data_root / "promise_nfr" / "nfr.csv"
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[("TF_Surface", r["id"])] = {
                    "input_text": r["RequirementText"],
                    "gold": r["class"],
                }
    # Mid
    p = data_root / "damir_pure" / "spans.jsonl"
    if p.exists():
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            out[("TF_Mid", r["id"])] = {
                "input_text": r["text"],
                "gold": r.get("spans", []),
                "gold_label": r.get("gold_label"),
            }
    # Reason
    p = data_root / "trace_etour" / "items.jsonl"
    if p.exists():
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            out[("TF_Reason", r["id"])] = {
                "input_text": r["source_text"],
                "gold": r.get("gold_target_ids", []),
                "n_candidates": (r.get("meta") or {}).get("n_candidates"),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=ROOT / "results" / "pillar1_fp16")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "coding" / "samples")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=SEED)
    # Soft quotas: Reason-heavy (coding critical path), then Surface, Mid.
    ap.add_argument("--reason-n", type=int, default=28)
    ap.add_argument("--surface-n", type=int, default=12)
    ap.add_argument("--mid-n", type=int, default=8)
    args = ap.parse_args()

    rows = _load_results(args.results_dir)
    fails = [r for r in rows if _fail_kind(r) not in ("ok", "unscored")]
    items = _load_item_text(args.data_root)

    by_task: dict[str, list] = defaultdict(list)
    for r in fails:
        by_task[r["_task"]].append(r)

    quotas = {
        "TF_Reason": args.reason_n,
        "TF_Surface": args.surface_n,
        "TF_Mid": args.mid_n,
    }
    # Clamp quotas to available + total n
    picked: list[dict] = []
    for task, want in quotas.items():
        pool = by_task.get(task, [])
        take = min(want, len(pool), max(0, args.n - len(picked)))
        if take <= 0:
            continue
        # Within task: stratify by model × fail_kind
        sample = stratified_sample(
            pool, take, args.seed,
            id_of=lambda r: f"{r['_model']}:{r['item_id']}:{r.get('run_uuid','')}",
            stratum_of=lambda r: f"{r['_model']}|{_fail_kind(r)}",
        )
        picked.extend(sample)

    # Stable global order
    picked = stratified_sample(
        picked, None, args.seed,
        id_of=lambda r: f"{r['_task']}:{r['_model']}:{r['item_id']}",
        stratum_of=None,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "wave1_manifest.csv"
    sheet = args.out_dir / "wave1_coding_sheet.csv"
    packets = args.out_dir / "wave1_packets.jsonl"
    readme = args.out_dir / "WAVE1_README.md"

    fieldnames = [
        "case_id", "task_id", "model_id", "quant", "item_id", "fail_kind",
        "quality", "malformed", "experiment_id", "source_file", "run_uuid",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(picked, 1):
            w.writerow({
                "case_id": f"W1-{i:03d}",
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

    sheet_fields = [
        "case_id", "task_id", "model_id", "quant", "item_id",
        "primary_code", "secondary_codes", "visibility", "quality_attribute",
        "rationale", "quote_span", "coder",
    ]
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sheet_fields)
        w.writeheader()
        for i, r in enumerate(picked, 1):
            w.writerow({
                "case_id": f"W1-{i:03d}",
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
            key = (r["_task"], r["item_id"])
            meta = items.get(key, {})
            raw = r.get("raw_model_output") or ""
            pkt = {
                "case_id": f"W1-{i:03d}",
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
                "raw_model_output": raw,
                "parsed_output": r.get("parsed_output"),
                "coding_hint": {
                    "TF_Reason": "Check JSON list of .java files vs gold_target_ids; note format drift vs wrong links.",
                    "TF_Surface": "Compare predicted NFR code vs gold class.",
                    "TF_Mid": "Compare predicted spans vs gold pronoun spans; empty-vs-nonempty matters.",
                }.get(r["_task"], ""),
            }
            fh.write(json.dumps(pkt, ensure_ascii=False) + "\n")

    # stratum counts
    from collections import Counter
    task_c = Counter(r["_task"] for r in picked)
    model_c = Counter(r["_model"] for r in picked)
    kind_c = Counter(_fail_kind(r) for r in picked)

    readme.write_text(
        f"""# Pillar 2 Wave 1 — coding packet

**Drawn:** seed={args.seed}, n={len(picked)} (target first wave; grow to 100–150 later).  
**Source:** `{args.results_dir.relative_to(ROOT)}` FP16 cells only (no quants yet).  
**Failure rule:** `quality < 1.0` (includes partial-credit Reason cases).

## Stratum counts
- By task: {dict(task_c)}
- By model: {dict(model_c)}
- By fail_kind: {dict(kind_c)}

## Files
| File | Use |
|------|-----|
| `wave1_manifest.csv` | Case index + quality/malformed flags |
| `wave1_coding_sheet.csv` | **Fill this** (codes stay empty until you code) |
| `wave1_packets.jsonl` | Full input / gold / raw output per case |
| `../codebook.md` | Mechanism + visibility codes (v0 sensitizing) |

## How to code (coder A)
1. Open `wave1_packets.jsonl` (one JSON object per line) and the matching row in `wave1_coding_sheet.csv`.
2. Read `input_text`, `gold`, and `raw_model_output`. Do **not** look at other models for the same item when deciding visibility.
3. Assign `primary_code` (M1–M4 or emergent M9-*), optional `secondary_codes`, and **required** `visibility` (V0/V1/V2).
4. Fill `quality_attribute` ∈ {{verifiability, unambiguity, traceability, normative-strength}}.
5. Short `rationale` + `quote_span` from the model output.
6. Leave blank rather than guess. Emergent mechanisms: use `M9-<short-name>` and add a definition note in the rationale.

## Blindness note
Sheet includes model_id for provenance. For a blinded pass, hide model/quant columns or use a redacted sheet; Wave 1 allows open coding for speed, with a later blinded double-code subset (30–50) for κ.

## Next waves
- Wave 2: add Q8/Q4 failures once quant ladder runs.
- Wave 3: top up Mid/Surface if taxonomy under-covers a family.
""",
        encoding="utf-8",
    )

    print(f"[ok] n={len(picked)} -> {args.out_dir}")
    print(f"  tasks={dict(task_c)}")
    print(f"  models={dict(model_c)}")
    print(f"  kinds={dict(kind_c)}")
    print(f"  sheet={sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
