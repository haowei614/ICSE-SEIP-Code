#!/usr/bin/env python3
"""Merge coderA + coderB sheets into kappa input CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coder-a", type=Path,
                    default=Path("coding/samples/wave1_coding_sheet.csv"))
    ap.add_argument("--coder-b", type=Path,
                    default=Path("coding/samples/wave1_coderB_sheet.csv"))
    ap.add_argument("--ids", type=Path,
                    default=Path("coding/samples/wave1_kappa_ids.txt"))
    ap.add_argument("--out", type=Path,
                    default=Path("coding/samples/wave1_double_coded.csv"))
    args = ap.parse_args()

    ids = {ln.strip() for ln in args.ids.read_text(encoding="utf-8").splitlines() if ln.strip()}
    a = {r["case_id"]: r for r in csv.DictReader(args.coder_a.open(encoding="utf-8"))}
    b = {r["case_id"]: r for r in csv.DictReader(args.coder_b.open(encoding="utf-8"))}

    rows = []
    missing_b = []
    for cid in sorted(ids):
        ra, rb = a.get(cid), b.get(cid)
        if not ra or not rb:
            continue
        if not (rb.get("primary_code") or "").strip() or not (rb.get("visibility") or "").strip():
            missing_b.append(cid)
            continue
        rows.append({
            "case_id": cid,
            "coder1": ra["primary_code"].strip(),
            "coder2": rb["primary_code"].strip(),
            "coder1_visibility": ra["visibility"].strip(),
            "coder2_visibility": rb["visibility"].strip(),
            # kappa.py expects coder1/coder2 — primary by default
        })

    if missing_b:
        print(f"[warn] coderB incomplete for {len(missing_b)} cases: {missing_b[:8]}...")
    if not rows:
        raise SystemExit("[abort] no complete double-coded rows.")

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "case_id", "coder1", "coder2",
            "coder1_visibility", "coder2_visibility",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {args.out} n={len(rows)}")
    print("Run primary κ:  python -m analysis.kappa", args.out)
    # also write visibility-only file
    vis_out = args.out.with_name(args.out.stem + "_visibility.csv")
    with vis_out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "coder1", "coder2"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "case_id": r["case_id"],
                "coder1": r["coder1_visibility"],
                "coder2": r["coder2_visibility"],
            })
    print(f"[ok] wrote {vis_out} (visibility κ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
