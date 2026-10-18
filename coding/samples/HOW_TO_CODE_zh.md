# Wave 1 怎么用、怎么填（中文）

你只需要改 **一个文件**：`wave1_coding_sheet.csv`。  
其他文件是**只读参考**，用来帮你看清「输入 / 金标 / 模型说了啥」。

---

## 1. 三个文件各干什么

| 文件 | 你要不要改 | 干什么 |
|------|------------|--------|
| **`wave1_coding_sheet.csv`** | **要改（主战场）** | 48 行待编码表；空白列由你填码 |
| **`wave1_packets.jsonl`** | 不改 | 每一行一个 JSON：含需求原文、金标、模型原文 |
| **`wave1_manifest.csv`** | 不改 | 索引：失败类型、quality、是否 malformed |
| `WAVE1_README.md` | 不改 | 英文抽样说明 |
| `../codebook.md` | 一般不改 | 机制码 M*、可见性 V* 的定义 |

建议用 Excel / Numbers / VS Code 打开 CSV；`packets.jsonl` 用编辑器按 `case_id` 搜索。

---

## 2. 推荐工作流（每条 1–2 分钟）

对 **同一条 `case_id`**（例如 `W1-001`）：

1. 在 `wave1_packets.jsonl` 里搜 `W1-001`，读这三块：
   - `input_text`：模型看到的需求/用例
   - `gold`：正确答案（链接文件名 / NFR 标签 / span）
   - `raw_model_output`：模型实际输出
2. 对照 `../codebook.md` 选码。
3. 在 `wave1_coding_sheet.csv` 找到同一 `case_id` 那一行，填空列。
4. 不确定 → **留空**，不要猜。

`task_id / model_id / quant / item_id` 已经填好，**不要改**。

---

## 3. 要填的列（从左到右）

| 列名 | 必填？ | 填什么 | 可选值 |
|------|--------|--------|--------|
| `primary_code` | **是** | 最主要的失败机制（只选一个） | `M1`…`M6` 或 `M9-短名` |
| `secondary_codes` | 否 | 次要机制，多个用 `;` 分隔 | 同上 |
| `visibility` | **是** | 这种错在真实审查里好不好发现 | `V0` / `V1` / `V2` |
| `quality_attribute` | 建议填 | 主要威胁的 RE 质量属性 | 见下 |
| `rationale` | **是** | 一句话：为什么是这个码 | 中英文均可 |
| `quote_span` | 建议填 | 从模型输出里摘一句作证据 | 短引语 |
| `coder` | 已填 | 你的代号 | 默认 `coderA`；第二编码者改成 `coderB` |

### `primary_code` 速查（Wave 1 最常用）

- **M5**：格式不对（散文、不是 JSON、标签乱写）→ 常配 **V0**
- **M6**：格式对，但内容错（错 Java 文件 / 错标签 / 错 span）→ 常配 **V1 或 V2**
- **M3**：长上下文后面才错（Reason 可考虑）
- **M1/M2/M4**：需求语义被改弱、漏隐含、术语不一致（少见也要标）
- **M9-xxx**：上面都不像，自己起名，在 `rationale` 里写定义

### `visibility` 怎么选（最重要）

假设审查员**没有金标**、每条大约 **60 秒**：

| 码 | 含义 | 典型情况 |
|----|------|----------|
| **V0** | 自动检查就能抓 | JSON 解析失败、`malformed=1`、明显不是要求的格式 |
| **V1** | 自动过了，但正常人扫一眼能觉得不对 | JSON 合法，但文件名和用例明显不搭 |
| **V2** | 自动过 + 正常快审也容易放过 | 看起来很像样，只有对照金标才知道错（**静默漂移**） |

判断技巧：`manifest` 里 `malformed=1` → 优先考虑 **V0**。  
格式正确但 `quality=0` → 在 **V1/V2** 里选：错得「一眼假」还是「像真的」。

### `quality_attribute` 可选

只填一个最贴的：

- `traceability` — 追错链（Reason 最多）
- `unambiguity` — 歧义/span（Mid）
- `verifiability` — 分类错导致难验证（Surface）
- `normative-strength` — shall/should 等被改弱（少见）

---

## 4. 完整示例：`W1-001`

**Packet 摘要**

- 任务：`TF_Reason`，模型：`gemma2_9b_it`
- 金标：`["ISearch.java"]`
- 模型输出：`["AdvancedSearch.java", "Search.java", "StandardSearch.java"]`（JSON 合法）
- quality=0，malformed=false → **形式对、链接错**

**sheet 可这样填：**

```text
primary_code        = M6
secondary_codes     =          （可空）
visibility          = V1
quality_attribute   = traceability
rationale           = Well-formed JSON list but none of the predicted files match gold ISearch.java; names look plausible for AdvancedSearch UC so easy to miss without gold.
quote_span          = ["AdvancedSearch.java", "Search.java", "StandardSearch.java"]
coder               = coderA
```

若你觉得「文件名和 AdvancedSearch 用例太像，60 秒审不过来」→ 可改成 **V2**（在 rationale 写清理由）。V1/V2 有争议很正常，第二编码者会再标。

---

## 5. 三种任务分别看什么

| 任务 | 金标在 packet 的哪 | 模型对了长什么样 | 常见码 |
|------|-------------------|------------------|--------|
| **TF_Reason** | `gold` = Java 文件名列表 | `["Foo.java", ...]` | M5 散文；M6 链错 |
| **TF_Surface** | `gold` = NFR 字母码（如 F, PE） | 只回一个码 | M5 乱码；M6 类别错 |
| **TF_Mid** | `gold` = span 列表；可能为空 | `[{"start":..,"end":..,"label":"anaphoric_ambiguity"}]` 或 `[]` | M5 非 JSON；M6 标错位置/乱标 |

---

## 6. 不要做的事

- 不要根据「这个模型平时挺强」去改码（就看本条输出）。
- 不要为了凑分布硬填；空着比乱填好。
- 不要改 `case_id` / `item_id` / 已有元数据列。
- 不要把成功的 case 当失败编（Wave 1 全是失败样本）。

---

## 7. 编完怎么交

保存 `wave1_coding_sheet.csv` 即可。  
第二编码者复制一份 sheet，把 `coder` 改成 `coderB`，只编抽中的 30–50 条子集（稍后另发名单）。

有问题优先标在 `rationale` 里写 `UNSURE:`，我们再对。
