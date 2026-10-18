#!/usr/bin/env python3
"""Long-form parser sensitivity table and Mid macros for the frozen parser.

Item-level percentile bootstrap, 10_000 resamples, numpy Generator seed 20260905,
2.5/97.5 percentiles. A macro resamples each model independently, then averages
the model means. Mid quality is the P3-A combined score under the named parser.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.rescore_parsers import (  # noqa: E402
    KIND,
    PARSERS,
    SOURCES,
    item_quality,
    load_jsonl,
    mean,
    refs_for,
)
import glob
from collections import defaultdict

SEED = 20260905
N_BOOT = 10_000
LABEL = {
    "lenient_unique": "unique",
    "lenient_unique_no_fence_skip": "unique_no_fence_skip",
    "lenient": "last",
    "lenient_first": "first",
    "legacy_greedy": "legacy_greedy",
}
MODELS = (
    "qwen25_7b_instruct",
    "mistral_7b_instruct",
    "llama31_8b_instruct",
    "gemma2_9b_it",
)


def macro_ci(vectors: list[np.ndarray]) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    point = float(np.mean([v.mean() for v in vectors]))
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = np.mean([
            v[rng.integers(0, len(v), len(v))].mean() for v in vectors
        ])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi)


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    os.chdir(root)
    references = refs_for("data")
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

    long_rows = []
    by_parser_quality = defaultdict(dict)
    mid_items = defaultdict(dict)
    for key, rows in sorted(cells.items()):
        source, task, model, precision = key
        if task not in KIND:
            continue
        cell = f"{source}|{task}|{model}|{precision}"
        for parser in PARSERS:
            qualities, malformed, ids = [], [], []
            for r in rows:
                ref = references.get(r.get("item_id"))
                if ref is None:
                    continue
                q, mal = item_quality(task, r.get("raw_model_output") or "", ref, parser)
                qualities.append(q)
                malformed.append(mal)
                ids.append(r.get("item_id"))
            well = [q for q, m in zip(qualities, malformed) if not m]
            label = LABEL[parser]
            long_rows.append({
                "cell": cell,
                "source": source,
                "task": task,
                "model": model,
                "precision": precision,
                "parser": label,
                "n": len(qualities),
                "malformed": int(sum(malformed)),
                "quality": f"{mean(qualities):.4f}",
                "q_ok": f"{mean(well):.4f}" if well else "",
                "n_wellformed": len(well),
            })
            by_parser_quality[(source, task, model, precision)][label] = mean(qualities)
            if task == "TF_Mid" and parser == "lenient_unique":
                mid_items[(source, model, precision)] = {
                    iid: q for iid, q in zip(ids, qualities)
                }

    out_dir = "analysis/out"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "parser_sensitivity.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)
    print(f"[wrote] {path} rows={len(long_rows)}")

    print("[diff] quality range across parsers > 0.05, aligned_v2 and FP16")
    seen = set()
    for (source, task, model, precision), quals in sorted(by_parser_quality.items()):
        if source not in ("aligned_v2", "transformers_fp16"):
            continue
        vals = [quals[LABEL[p]] for p in PARSERS]
        span = max(vals) - min(vals)
        if span > 0.05:
            seen.add((model, task))
            print(f"  {source}|{task}|{model}|{precision} range={span:.4f} "
                  + " ".join(f"{LABEL[p]}={quals[LABEL[p]]:.4f}" for p in PARSERS))
    print("[diff] model-task pairs:", ", ".join(f"{m}|{t}" for m, t in sorted(seen)) or "(none)")

    print("[macro] Mid combined under unique")
    specs = [("transformers_fp16", "FP16", "FP16")]
    for prec in ("F16", "Q8_0", "Q6_K", "Q4_K_M"):
        specs.append(("aligned_v2", prec, prec))
    for source, precision, label in specs:
        for drop_llama, tag in ((False, "with_llama"), (True, "without_llama")):
            vectors = []
            used = []
            for model in MODELS:
                if drop_llama and model == "llama31_8b_instruct":
                    continue
                items = mid_items.get((source, model, precision))
                if not items:
                    continue
                vectors.append(np.array(list(items.values()), dtype=float))
                used.append(model)
            if len(vectors) < 2:
                print(f"  {label} {tag}: missing models {used}")
                continue
            point, lo, hi = macro_ci(vectors)
            per = ", ".join(f"{m}={v.mean():.4f}" for m, v in zip(used, vectors))
            print(f"  {label} {tag} n_models={len(used)} macro={point:.4f} "
                  f"CI=({lo:.4f}, {hi:.4f}) [{per}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
