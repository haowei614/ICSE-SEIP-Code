"""Cohen's kappa for the Pillar-2 second-coder subset.

Input: a CSV with columns item_id, coder1, coder2 (category labels) for the
30-50 case double-coded subset. Outputs kappa and Landis-Koch interpretation.

Usage: python -m analysis.kappa coding/double_coded.csv
"""
from __future__ import annotations

import argparse
import csv


def landis_koch(k: float) -> str:
    if k < 0.00:
        return "poor"
    if k <= 0.20:
        return "slight"
    if k <= 0.40:
        return "fair"
    if k <= 0.60:
        return "moderate"
    if k <= 0.80:
        return "substantial"
    return "almost perfect"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args(argv)

    c1, c2 = [], []
    with open(args.csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            c1.append(row["coder1"].strip())
            c2.append(row["coder2"].strip())

    if not c1:
        raise SystemExit("[abort] no double-coded rows found.")

    try:
        from sklearn.metrics import cohen_kappa_score
        k = cohen_kappa_score(c1, c2)
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"[abort] install scikit-learn: {e}")

    print(f"n={len(c1)}  kappa={k:.3f}  ({landis_koch(k)} agreement)")
    print("Paste into macros.tex: \\renewcommand{\\CohenKappa}{" f"{k:.2f}" "}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
