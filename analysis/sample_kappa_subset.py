#!/usr/bin/env python3
"""Draw a deterministic second-coder (κ) subset from Wave-1 coded sheet.

Stratifies by task × visibility (the acceptance-critical axis) and writes:
  coding/samples/wave1_coderB_sheet.csv  — blank codes for coderB
  coding/samples/wave1_kappa_ids.txt     — case_id list
  coding/samples/KAPPA_HOWTO_zh.md       — instructions
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sampling import stratified_sample  # noqa: E402

SEED = 20260907
DEFAULT_N = 36  # within 30–50 target


def main() -> int:
    sheet_path = ROOT / "coding" / "samples" / "wave1_coding_sheet.csv"
    out_dir = ROOT / "coding" / "samples"
    rows = list(csv.DictReader(sheet_path.open(encoding="utf-8")))
    # only fully coded by coderA
    rows = [r for r in rows if (r.get("primary_code") or "").strip()
            and (r.get("visibility") or "").strip()]

    n = min(DEFAULT_N, len(rows))
    picked = stratified_sample(
        rows, n, SEED,
        id_of=lambda r: r["case_id"],
        stratum_of=lambda r: f"{r['task_id']}|{(r.get('visibility') or '').strip()}",
    )

    # coderB sheet: same metadata, empty judgment columns
    fields = list(rows[0].keys())
    out_sheet = out_dir / "wave1_coderB_sheet.csv"
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

    ids_path = out_dir / "wave1_kappa_ids.txt"
    ids_path.write_text("\n".join(r["case_id"] for r in picked) + "\n", encoding="utf-8")

    # gold-blind packets subset pointer
    (out_dir / "KAPPA_HOWTO_zh.md").write_text(
        f"""# 第二编码者（κ）怎么做

**子集：** n={len(picked)}，seed={SEED}，分层：`task × visibility`（按 coderA 的可见性分层抽样，保证 V0/V1/V2 都有）。

## 文件
| 文件 | 用途 |
|------|------|
| `wave1_coderB_sheet.csv` | **coderB 填写**（判断列为空） |
| `wave1_kappa_ids.txt` | 本案 ID 列表 |
| `wave1_packets.jsonl` | 仍用完整 packet；只编 ids 里的 case |
| `../codebook.md` | 与 coderA 相同 |

## 规则
1. **不要看** `wave1_coding_sheet.csv`（coderA 答案）。
2. 对每个 `case_id`：在 `wave1_packets.jsonl` 读输入/金标/模型输出，填 `wave1_coderB_sheet.csv`。
3. 必填：`primary_code`、`visibility`、`rationale`（建议填 `quality_attribute`、`quote_span`）。
4. 与 Wave 1 相同码表（M5/M6/… 与 V0/V1/V2）。

## 算 κ（编完后）
合并两列后运行：

```bash
python -m analysis.kappa coding/samples/wave1_double_coded.csv
```

`wave1_double_coded.csv` 格式：`case_id,coder1_primary,coder2_primary,coder1_visibility,coder2_visibility`
（可用脚本从两张 sheet 自动生成，见 `analysis/merge_double_coded.py`。）
""",
        encoding="utf-8",
    )

    from collections import Counter
    print(f"[ok] kappa subset n={len(picked)}")
    print("  tasks", dict(Counter(r["task_id"] for r in picked)))
    print("  vis(A)", dict(Counter(r["visibility"].strip() for r in picked)))
    print("  ->", out_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
