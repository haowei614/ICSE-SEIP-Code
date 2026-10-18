"""Visible-vs-silent tallies and normal-vs-standards-aware comparison (Pillar 2, T6).

Detection channels (paper §5.3):
  Schema    automated checks (schema/format/consistency)
  Unaided   realistic time-budgeted human review (fixed budget per case)
  Checklist standards-aware / enhanced review (verifiability, unambiguity,
            traceability, normative strength checklist)

Definitions:
  visible failure       = caught by Schema OR Unaided
  silent semantic drift = semantically wrong AND passes Schema AND not caught by Unaided
                          ("silent" != undetectable under unlimited inspection)

This script only TALLIES human-produced review logs. It never decides correctness
itself and never fabricates review outcomes: if the log is missing or incomplete,
it aborts.

Expected input CSV (one row per reviewed case), produced by the review sessions:
  case_id, task_id, model_id, quant, reviewer,
  semantically_incorrect (0/1),   # ground-truth-based, filled after review
  c1_flagged (0/1),               # automated check flagged it
  c2_flagged (0/1),               # normal time-budgeted review flagged it
  c3_flagged (0/1 or blank),      # standards-aware review flagged it (optional)
  review_seconds, order_index     # for fatigue/order-effect diagnostics

Usage:
  python -m analysis.human_review coding/human_review.csv --out analysis/out
"""
from __future__ import annotations

import argparse
import csv
import os

REQUIRED = [
    "case_id", "task_id", "semantically_incorrect",
    "c1_flagged", "c2_flagged",
]


def _to_int(v):
    v = (v or "").strip()
    if v == "":
        return None
    return int(float(v))


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("[abort] no review rows found.")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise SystemExit(f"[abort] review CSV missing columns: {missing}")
    return rows


def tally(rows: list[dict]) -> dict:
    visible = silent = incorrect = 0
    c2_escape = c3_escape = c3_scored = 0
    for r in rows:
        inc = _to_int(r["semantically_incorrect"])
        c1 = _to_int(r["c1_flagged"])
        c2 = _to_int(r["c2_flagged"])
        c3 = _to_int(r.get("c3_flagged"))
        if None in (inc, c1, c2):
            raise SystemExit(
                f"[abort] incomplete review row {r.get('case_id')}: fill "
                "semantically_incorrect, c1_flagged, c2_flagged before tallying."
            )
        if inc:
            incorrect += 1
        if c1 or c2:
            visible += 1
        # silent = wrong, passes Schema, not caught by Unaided review
        is_silent = bool(inc) and not c1 and not c2
        if is_silent:
            silent += 1
        # escape-rate comparison over wrong cases that pass Schema
        if inc and not c1:
            c2_escape += 0 if c2 else 1
            if c3 is not None:
                c3_scored += 1
                c3_escape += 0 if c3 else 1
    return {
        "n": len(rows), "incorrect": incorrect,
        "visible": visible, "silent": silent,
        "silent_escape_normal": c2_escape,
        "silent_escape_std_aware": c3_escape if c3_scored else None,
        "c3_scored": c3_scored,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args(argv)

    rows = load_rows(args.csv)
    t = tally(rows)
    os.makedirs(args.out, exist_ok=True)
    print(f"n={t['n']}  incorrect={t['incorrect']}  "
          f"visible={t['visible']}  silent={t['silent']}")
    print(f"silent-drift escape (Unaided)        = {t['silent_escape_normal']}")
    if t["silent_escape_std_aware"] is None:
        print("silent-drift escape (Checklist)      = [not scored: fill c3_flagged]")
    else:
        print(f"silent-drift escape (Checklist)      = {t['silent_escape_std_aware']}"
              f"  (over {t['c3_scored']} scored)")
    print("[note] map these to \\VisibleFailRate, \\SilentDriftRate, "
          "\\SilentDriftRateNormal, \\SilentDriftRateStdAware after human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
