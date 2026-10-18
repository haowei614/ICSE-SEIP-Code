#!/usr/bin/env python3
"""Draw a deterministic second-coder (κ) subset from Wave-2 coded sheet.

Mirrors sample_kappa_subset.py (Wave 1). Stratifies by task × visibility.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sampling import stratified_sample  # noqa: E402

SEED = 20260907
DEFAULT_N = 36


def main() -> int:
    sheet_path = ROOT / "coding" / "samples" / "wave2_coding_sheet.csv"
    out_dir = ROOT / "coding" / "samples"
    rows = list(csv.DictReader(sheet_path.open(encoding="utf-8")))
    rows = [r for r in rows if (r.get("primary_code") or "").strip()
            and (r.get("visibility") or "").strip()]

    n = min(DEFAULT_N, len(rows))
    picked = stratified_sample(
        rows, n, SEED,
        id_of=lambda r: r["case_id"],
        stratum_of=lambda r: f"{r['task_id']}|{(r.get('visibility') or '').strip()}",
    )

    fields = list(rows[0].keys())
    out_sheet = out_dir / "wave2_coderB_sheet.csv"
    with out_sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in picked:
            w.writerow({
                "case_id": r["case_id"],
                "task_id": r["task_id"],
                "model_id": r["model_id"],
                "quant": r["quant"],
                "item_id": r["item_id"],
                "primary_code": "",
                "secondary_codes": "",
                "visibility": "",
                "quality_attribute": "",
                "rationale": "",
                "quote_span": "",
                "coder": "coderB",
            })

    ids_path = out_dir / "wave2_kappa_ids.txt"
    ids_path.write_text("\n".join(r["case_id"] for r in picked) + "\n", encoding="utf-8")

    (out_dir / "WAVE2_KAPPA_HOWTO_zh.md").write_text(
        f"""# Wave 2 第二编码者（κ）怎么做

**子集：** n={len(picked)}，seed={SEED}，分层：`task × visibility`（按 coderA 可见性分层）。

## 文件
| 文件 | 用途 |
|------|------|
| `wave2_coderB_sheet.csv` | **coderB 填写** |
| `wave2_kappa_ids.txt` | 本案 ID |
| `wave2_packets.jsonl` | packet；只编 ids 里的 case |
| `../codebook.md` | codebook **v1**（与 coderA 相同） |

## 规则
1. **不要看** `wave2_coding_sheet.csv`（coderA）。
2. 对每个 case：读 packet，填 `wave2_coderB_sheet.csv`。
3. 必填：`primary_code`、`visibility`、`rationale`。
4. 注意量化失败里 **M5 会更多**；但码表规则不变：M5⇔V0。

## 算 κ
```bash
cd replication
python analysis/merge_double_coded.py \\
  --coder-a coding/samples/wave2_coding_sheet.csv \\
  --coder-b coding/samples/wave2_coderB_sheet.csv \\
  --ids coding/samples/wave2_kappa_ids.txt \\
  --out coding/samples/wave2_double_coded.csv
python -m analysis.kappa coding/samples/wave2_double_coded.csv
python -m analysis.kappa coding/samples/wave2_double_coded_visibility.csv
```
""",
        encoding="utf-8",
    )

    print(f"[ok] wave2 kappa subset n={len(picked)}")
    print("  tasks", dict(Counter(r["task_id"] for r in picked)))
    print("  vis(A)", dict(Counter(r["visibility"].strip() for r in picked)))
    print("  primary(A)", dict(Counter(r["primary_code"].strip() for r in picked)))
    print("  ->", out_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
