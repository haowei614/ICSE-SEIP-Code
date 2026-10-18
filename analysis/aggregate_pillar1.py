#!/usr/bin/env python3
"""Aggregate Pillar-1 FP16 + quant cells; also report well-formed-only quality.

Mid quant means are inflated when malformed→empty pred matches empty gold.
`quality_mean_ok` averages quality only over non-malformed rows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def load_dirs(paths: list[str]) -> list[dict]:
    recs = []
    for path in paths:
        for fp in glob.glob(os.path.join(path, "*.jsonl")):
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dirs", nargs="+")
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args()

    recs = load_dirs(args.results_dirs)
    if any(r.get("mock") for r in recs):
        raise SystemExit("[integrity] mock records present; refuse.")
    if not recs:
        raise SystemExit("[abort] no records")

    import pandas as pd

    df = pd.DataFrame(recs)
    for col in ("malformed_output", "oom", "timeout"):
        if col not in df.columns:
            df[col] = False
    df["malformed_output"] = df["malformed_output"].fillna(False).astype(bool)

    keys = ["task_family", "model_id", "precision"]
    rows = []
    for key, g in df.groupby(keys):
        ok = g[~g["malformed_output"]]
        rows.append({
            "task_family": key[0],
            "model_id": key[1],
            "precision": key[2],
            "n": len(g),
            "quality_mean": g["quality"].mean(),
            "quality_mean_ok": ok["quality"].mean() if len(ok) else None,
            "n_ok": int(len(ok)),
            "malformed_rate": float(g["malformed_output"].mean()),
            "tokens_mean": g["output_tokens"].mean(),
            "latency_mean": g["total_latency_s"].mean(),
            "peak_memory_mb": g["peak_memory_mb"].max(),
            "oom_count": int(g["oom"].sum()),
        })
    out = pd.DataFrame(rows).sort_values(keys)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.out, "pillar1_cells.csv")
    out.to_csv(csv_path, index=False)
    print(f"[ok] wrote {csv_path} ({len(out)} cells from {len(recs)} records)")
    # wide summary for paper glance
    pivot = out.pivot_table(
        index=["task_family", "model_id"],
        columns="precision",
        values=["quality_mean", "malformed_rate"],
    )
    pivot_path = os.path.join(args.out, "pillar1_pivot.csv")
    pivot.to_csv(pivot_path)
    print(f"[ok] wrote {pivot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
