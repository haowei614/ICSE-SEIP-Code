# Wave 2 第二编码者（κ）怎么做

**子集：** n=36，seed=20260907，分层：`task × visibility`（按 coderA 可见性分层）。

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
python analysis/merge_double_coded.py \
  --coder-a coding/samples/wave2_coding_sheet.csv \
  --coder-b coding/samples/wave2_coderB_sheet.csv \
  --ids coding/samples/wave2_kappa_ids.txt \
  --out coding/samples/wave2_double_coded.csv
python -m analysis.kappa coding/samples/wave2_double_coded.csv
python -m analysis.kappa coding/samples/wave2_double_coded_visibility.csv
```
