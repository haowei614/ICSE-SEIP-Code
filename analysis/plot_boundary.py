"""Figures F1/F2: quality vs quantization faceted by task family, + resource overlay.

Reads analysis/out/pillar1_cells.csv (produced by aggregate.py from REAL runs).
Draws nothing if quality columns are empty. Never invents points.

Usage: python -m analysis.plot_boundary analysis/out/pillar1_cells.csv --out analysis/out
"""
from __future__ import annotations

import argparse
import os


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args(argv)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv)
    if df.empty or df["quality_mean"].dropna().empty:
        raise SystemExit(
            "[abort] no real quality values to plot. Fill from real runs first."
        )

    os.makedirs(args.out, exist_ok=True)
    quant_order = ["Q4_K_M", "Q6_K", "Q8_0", "FP16", "BF16"]
    df["precision"] = pd.Categorical(df["precision"], quant_order, ordered=True)

    # F1: quality vs precision per task family (one line per model).
    for task_family, g in df.groupby("task_family"):
        fig, ax = plt.subplots(figsize=(4, 3))
        for model_id, gm in g.groupby("model_id"):
            gm = gm.sort_values("precision")
            ax.plot(gm["precision"].astype(str), gm["quality_mean"], marker="o", label=model_id)
        ax.set_title(f"{task_family}: quality vs precision")
        ax.set_xlabel("precision"); ax.set_ylabel("quality (locked basis)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fp = os.path.join(args.out, f"F1_{task_family}.pdf")
        fig.savefig(fp); plt.close(fig)
        print(f"[ok] {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
