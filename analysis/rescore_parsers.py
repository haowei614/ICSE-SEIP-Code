#!/usr/bin/env python3
"""Rescore stored cells with lenient_unique as the primary parser.

Mid quality is the P3-A combined score. Surface and Reason quality are the
existing item scores. only_json_compliance is the share of Mid and Reason
items whose whole output is schema-valid JSON; it is not a quality score.
Surface does not use the JSON parsers.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import distinct_schema_values, only_json_compliant, score  # noqa: E402
from src import tasks as T  # noqa: E402

PARSERS = (
    "lenient_unique",
    "lenient_unique_no_fence_skip",
    "lenient",
    "lenient_first",
    "legacy_greedy",
)
KIND = {
    "TF_Surface": "classification",
    "TF_Mid": "span_detection",
    "TF_Reason": "trace_link",
}
SOURCES = [
    ("results/pillar1_fp16", "transformers_fp16"),
    ("results/pillar1_fp16_v2", "transformers_fp16_v2"),
    ("results/pillar1_aligned_v2", "aligned_v2"),
    ("results/pillar1_grammar_v2", "grammar_v2"),
    ("results/pillar1_f16_gguf_aligned", "superseded_string_256"),
    ("results/pillar1_quant_aligned", "superseded_string_256"),
    ("results/pillar1_quant", "legacy_raw"),
    ("results/pillar1_f16_gguf", "legacy_raw"),
]


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def refs_for(data_root: str) -> dict[str, dict]:
    out = {}
    for cfg in (
        {"kind": "classification", "dataset": "promise_nfr", "long_context": False},
        {"kind": "span_detection", "dataset": "damir_pure", "long_context": False},
        {"kind": "trace_link", "dataset": "trace_etour", "long_context": True},
    ):
        for it in T.load_task(cfg, data_root=data_root):
            out[it.item_id] = it.reference
    locked = os.path.join(data_root, "trace_etour_locked")
    if os.path.isdir(locked):
        for it in T.load_task(
            {"kind": "trace_link", "dataset": "trace_etour_locked", "long_context": True},
            data_root=data_root,
        ):
            out[it.item_id] = it.reference
    return out


def item_quality(task: str, raw: str, ref: dict, parser: str) -> tuple[float, bool]:
    kind = KIND[task]
    mid = "p3a" if task == "TF_Mid" else "legacy"
    q, detail = score("A", kind, raw, ref, mid_metric=mid, parser=parser)
    if task == "TF_Mid":
        return float(detail["combined"]), bool(detail["malformed"])
    return float(q), bool(detail["malformed"])


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    os.chdir(root)
    data_root = "data"
    references = refs_for(data_root)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for rel, source in SOURCES:
        if not os.path.isdir(rel):
            print(f"[skip] missing {rel}")
            continue
        for fp in sorted(glob.glob(os.path.join(rel, "*.jsonl"))):
            rows = load_jsonl(fp)
            if not rows or any(r.get("mock") for r in rows):
                continue
            key = (source, rows[0].get("task_family"), rows[0].get("model_id"),
                   rows[0].get("precision"))
            cells[key].extend(rows)

    table = []
    for key, rows in sorted(cells.items()):
        source, task, model, precision = key
        if task not in KIND:
            continue
        stats = {}
        for parser in PARSERS:
            qualities, malformed = [], []
            for r in rows:
                ref = references.get(r.get("item_id"))
                if ref is None:
                    continue
                q, mal = item_quality(task, r.get("raw_model_output") or "", ref, parser)
                qualities.append(q)
                malformed.append(mal)
            n_mal = int(sum(malformed))
            well = [q for q, m in zip(qualities, malformed) if not m]
            stats[parser] = {
                "n": len(qualities),
                "malformed": n_mal,
                "quality": mean(qualities),
                "q_ok": mean(well),
                "n_wellformed": len(well),
            }
        n_json = 0
        n_ambiguous = 0
        n_value_diff = 0
        n_scored = stats["lenient_unique"]["n"]
        if task in ("TF_Mid", "TF_Reason"):
            kind = KIND[task]
            for r in rows:
                if references.get(r.get("item_id")) is None:
                    continue
                raw = r.get("raw_model_output") or ""
                if only_json_compliant(raw, kind):
                    n_json += 1
                values = distinct_schema_values(raw, kind, skip_code_fences=True)
                if len(values) >= 2:
                    n_ambiguous += 1
                last = score("A", kind, raw, references[r["item_id"]],
                             mid_metric="p3a" if task == "TF_Mid" else "legacy",
                             parser="lenient")
                uniq = score("A", kind, raw, references[r["item_id"]],
                             mid_metric="p3a" if task == "TF_Mid" else "legacy",
                             parser="lenient_unique")
                if abs(float(uniq[0]) - float(last[0])) > 1e-9 or uniq[1]["malformed"] != last[1]["malformed"]:
                    n_value_diff += 1
            compliance = n_json / n_scored if n_scored else float("nan")
        else:
            compliance = None
        row = {
            "cell": f"{source}|{task}|{model}|{precision}",
            "n": n_scored,
        }
        for parser in PARSERS:
            row[f"malformed_{parser}"] = stats[parser]["malformed"]
        for parser in ("lenient_unique", "lenient", "lenient_first"):
            s = stats[parser]
            row[f"quality_{parser}"] = f"{s['quality']:.4f}"
            row[f"q_ok_{parser}"] = f"{s['q_ok']:.4f}" if s["n_wellformed"] else ""
        row["n_wellformed"] = stats["lenient_unique"]["n_wellformed"]
        row["n_ambiguous"] = "" if compliance is None else n_ambiguous
        row["only_json_compliance"] = "" if compliance is None else f"{compliance:.4f}"
        row["unique_vs_last_items"] = n_value_diff
        table.append(row)

    out_dir = "analysis/out"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "parser_rescore.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"[wrote] {path} rows={len(table)}")
    print("cell | mal unique/nofence/last/first/legacy | quality unique/last/first | q_ok | n_well | n_amb | only_json")
    for row in table:
        flag = " **" if int(row["unique_vs_last_items"]) > 2 else ""
        print(
            f"{row['cell']} | "
            f"{row['malformed_lenient_unique']}/{row['malformed_lenient_unique_no_fence_skip']}/"
            f"{row['malformed_lenient']}/{row['malformed_lenient_first']}/{row['malformed_legacy_greedy']} | "
            f"{row['quality_lenient_unique']}/{row['quality_lenient']}/{row['quality_lenient_first']} | "
            f"{row['q_ok_lenient_unique']}/{row['q_ok_lenient']}/{row['q_ok_lenient_first']} | "
            f"{row['n_wellformed']} | {row['n_ambiguous']} | {row['only_json_compliance']}{flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
