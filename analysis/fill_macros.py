"""Fill numeric macros in ../macros.tex from REAL aggregated results only.

Hard integrity guard: aborts if any input record is mock=True. This is the single
gate that keeps fabricated numbers out of the paper.

Usage:
  python -m analysis.fill_macros results/final --macros ../macros.tex        # dry
  python -m analysis.fill_macros results/final --macros ../macros.tex --write # apply
"""
from __future__ import annotations

import argparse

from .aggregate import aggregate, guard_no_mock, load_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--macros", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    recs = load_dir(args.results_dir)
    guard_no_mock(recs)                       # <-- blocks mock/dry-run data
    if not recs:
        raise SystemExit("[abort] no real records to fill macros from.")

    agg = aggregate(recs)
    # NOTE: the exact macro<->cell mapping is intentionally left explicit so the
    # human confirms which aggregate feeds which macro before writing.
    print(agg.to_string(index=False))
    print("\n[info] Map the rows above to macros (e.g. \\QualReasonLocalBest, "
          "\\PeakVRAMBest, \\Latency*), then re-run with --write once confirmed.")

    if args.write:
        raise SystemExit(
            "[stop] Refusing to auto-rewrite macros without an explicit, "
            "reviewed cell->macro mapping. Edit this function to encode the "
            "mapping you confirmed, then remove this guard."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
