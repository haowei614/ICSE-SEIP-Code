# 第二编码者（κ）怎么做

**子集：** n=36，seed=20260907，分层：`task × visibility`（按 coderA 的可见性分层抽样，保证 V0/V1/V2 都有）。

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

```bash
cd replication
python analysis/merge_double_coded.py
python -m analysis.kappa coding/samples/wave1_double_coded.csv
python -m analysis.kappa coding/samples/wave1_double_coded_visibility.csv
```

`merge_double_coded.py` 会跳过 coderB 未填完的行。
