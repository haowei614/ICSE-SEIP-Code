"""Normalized quality retention + pre-specified boundary detection (Pillar 1, T2).

QualityRetention(m, q, t) = Quality(m, q, t) / Quality(m, ref_local(m), t)

where ref_local(m) is the SAME model's highest-precision LOCAL reference (see
config `reference_precision` / per-model overrides). Retention isolates the extra
loss attributable to quantization from the model's intrinsic capability, and is
used ONLY for cross-task degradation-SHAPE comparison. Raw task quality remains
the primary task-level metric and is never replaced by retention.

Boundary detection is PRE-SPECIFIED (lock parameters before inspecting aggregates):
segmented (piecewise-linear) regression over the ordered precision axis plus a
change-point test, with bootstrap confidence intervals on the break location. A
"cliff" is reported ONLY if the pre-registered discontinuity criterion is met;
otherwise degradation is reported as gradual.

Integrity: consumes ONLY real aggregated cells (aggregate.py already blocks mock
records). Produces nothing if quality columns are empty; never invents points.

Usage:
  python -m analysis.retention analysis/out/pillar1_cells.csv --out analysis/out
"""
from __future__ import annotations

import argparse
import os

# Ordered from LOWEST to HIGHEST precision. The highest present precision per
# (model, task) is used as ref_local for retention. Extend if BF16 is emitted.
PRECISION_ORDER = ["Q4_K_M", "Q6_K", "Q8_0", "FP16", "BF16"]

# --- PRE-REGISTRATION TODO (lock before inspecting aggregates) --------------
# Fill these from config/experiment.yaml:boundary_detection BEFORE the main runs.
# They are intentionally None so the script refuses to declare a cliff until a
# human has committed to a criterion.
CHANGEPOINT_CONFIG = {
    "method": "segmented_regression+changepoint",   # documented in paper §3.4
    "min_segment_points": None,   # TODO: e.g. 2
    "n_bootstrap": None,          # TODO: e.g. 1000
    "alpha": None,                # TODO: e.g. 0.05
    # Discontinuity criterion, e.g. "slope_ratio>=K and CI_nonoverlap".
    "cliff_criterion": None,      # TODO: pre-register exact rule
}


def _precision_rank(q: str) -> int:
    return PRECISION_ORDER.index(q) if q in PRECISION_ORDER else -1


def compute_retention(df):
    """Add a `retention` column normalized against each model's highest-precision
    local reference, per (task_family, model_id). Returns a new DataFrame."""
    import pandas as pd  # noqa: F401

    out = df.copy()
    out["_rank"] = out["precision"].map(_precision_rank)
    out["retention"] = float("nan")
    for (task_family, model_id), g in out.groupby(["task_family", "model_id"]):
        g_valid = g[g["quality_mean"].notna() & (g["_rank"] >= 0)]
        if g_valid.empty:
            continue
        ref_row = g_valid.loc[g_valid["_rank"].idxmax()]
        ref_q = ref_row["quality_mean"]
        if ref_q is None or ref_q == 0 or ref_q != ref_q:  # None/0/NaN
            # Cannot normalize against a zero/undefined local reference.
            continue
        idx = g_valid.index
        out.loc[idx, "retention"] = out.loc[idx, "quality_mean"] / ref_q
        out.loc[idx, "ref_precision"] = ref_row["precision"]
    return out.drop(columns=["_rank"])


def detect_boundary(_task_df):
    """Pre-specified change-point / segmented-regression boundary detection.

    Refuses to run until CHANGEPOINT_CONFIG is fully pre-registered, so no cliff
    can be declared on an unlocked, post-hoc criterion.
    """
    missing = [k for k, v in CHANGEPOINT_CONFIG.items() if v is None]
    if missing:
        raise SystemExit(
            "[abort] boundary detection is not pre-registered. Set these keys in "
            f"CHANGEPOINT_CONFIG (and config/experiment.yaml): {missing}. "
            "A cliff must not be declared on a post-hoc criterion."
        )
    # Implementation intentionally deferred until the criterion is locked.
    raise NotImplementedError(
        "Implement segmented regression + change-point test with bootstrap CIs "
        "using the pre-registered criterion. Report 'cliff' iff the criterion is "
        "met; otherwise report 'gradual'."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="analysis/out/pillar1_cells.csv from aggregate.py")
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args(argv)

    import pandas as pd

    df = pd.read_csv(args.csv)
    if df.empty or df["quality_mean"].dropna().empty:
        raise SystemExit(
            "[abort] no real quality values. Fill from real runs (aggregate.py) first."
        )

    ret = compute_retention(df)
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "pillar1_retention.csv")
    ret.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv} ({ret['retention'].notna().sum()} normalized cells)")
    print("[next] pre-register CHANGEPOINT_CONFIG, then run detect_boundary per task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
