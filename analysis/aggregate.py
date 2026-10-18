"""Aggregate run records into Pillar-1 tables (T2/T3 inputs).

Refuses to aggregate mock records into anything the paper consumes.
Usage: python -m analysis.aggregate results/final --out analysis/out
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load_dir(path: str) -> list[dict]:
    recs: list[dict] = []
    for fp in glob.glob(os.path.join(path, "*.jsonl")):
        with open(fp, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def guard_no_mock(recs: list[dict]) -> None:
    if any(r.get("mock") for r in recs):
        raise SystemExit(
            "[integrity] Mock records present. Aggregation for the paper is "
            "blocked. Run a real backend on GPU hardware first."
        )


def aggregate(recs: list[dict]):
    import pandas as pd

    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("[abort] no records found.")
    # NOTE: quality_mean = corpus mean of per-item quality (classification->accuracy,
    # span/trace->mean F1). For macro-F1 per class, aggregate quality_detail instead.
    for col in ("malformed_output", "oom", "timeout"):
        if col not in df:
            df[col] = False
    keys = ["task_family", "model_id", "precision"]
    agg = df.groupby(keys).agg(
        n=("item_id", "count"),
        quality_mean=("quality", "mean"),
        tokens_mean=("output_tokens", "mean"),
        latency_mean=("total_latency_s", "mean"),
        ttft_mean=("ttft_s", "mean"),
        tokens_per_s_mean=("tokens_per_s", "mean"),
        peak_memory_mb=("peak_memory_mb", "max"),
        malformed_rate=("malformed_output", "mean"),
        oom_count=("oom", "sum"),
        timeout_count=("timeout", "sum"),
    ).reset_index()
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args(argv)

    recs = load_dir(args.results_dir)
    guard_no_mock(recs)
    agg = aggregate(recs)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "pillar1_cells.csv")
    agg.to_csv(csv_path, index=False)
    print(f"[ok] wrote {csv_path} ({len(agg)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
