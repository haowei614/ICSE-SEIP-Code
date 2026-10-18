# P1 摸底汇报：GGUF 请求路径与 schema（动手前）

Date: 2026-09-07 · Status: **investigation only**（尚未改代码、未重跑）

## 1. GGUF 侧如何发起请求

`replication/src/backends.py` → `LlamaCppBackend.generate()`：

- 加载：`llama_cpp.Llama(model_path=gguf, n_gpu_layers=-1, n_ctx=8192, n_batch=512)`
- 调用：`self._llm(prompt, max_tokens=..., temperature=0, top_p=1, seed=..., stream=True)`
- **当前无** `grammar` / `json_schema` / `response_format` 参数 → **完全自由解码**
- FP16 走另一条路径：`TransformersBackend` + `model.generate(...)`，同样无约束解码

结论：FP16 vs GGUF 的 malformed 差距，**混杂了「量化」与「llama.cpp 自由生成 JSON」两条通道**——与 P1 prompt 诊断一致。

## 2. 输出 schema 在哪里定义

| 层 | 位置 | 内容 |
|----|------|------|
| Prompt 契约 | `tasks.py:build_prompt` (`PROMPT_VERSION=p3`) | Surface: 单 label token；Mid: JSON list of `{start,end,label}` 或 `[]`；Reason: JSON list of `*.java` 文件名 |
| 解析/判 malformed | `metrics.py:_extract_label` / `_extract_json` | parse 失败 → `malformed=True` |
| 记录 | `schema.py` / `RunRecord.malformed_output` | 不丢弃失败 run |

**没有**单独的 JSON Schema 文件；schema 是 prompt 文字 + 解析器约定。

## 3. Grammar 设计要点（下一步实现）

与现有解析器对齐：

1. **TF-Surface**：枚举 `F|A|L|LF|MN|O|PE|SC|SE|US|FT|PO`（及大小写变体策略：grammar 固定大写，解析已 `.lower()`）
2. **TF-Mid**：`[]` 或 `[{...},...]`，字段 `start`/`end` int，`label` 固定 `"anaphoric_ambiguity"`
3. **TF-Reason**：`["Foo.java", ...]` 字符串数组

llama-cpp-python：`Llama(...,)` 生成时传 `grammar=LlamaGrammar.from_string(gbnf)`。

## 4. 计划中的配置维度（未实施）

- `decoding_mode ∈ {free, grammar}`
- 结果目录：`results/pillar1_grammar/`（**不覆盖** `pillar1_fp16/` / `pillar1_quant/`）
- 冻结：p3、temp 0、seed、n_ctx、n_batch、max_tokens、n=40 分层样本

## 5. 阻塞

完整 4×3×3 grammar 重跑需要 **GPU 节点**（aisin-repair）。本机仓库可先合入 grammar 代码与 dry-run；上机跑完再出对照表。

---

下一步：等你确认后实现 GBNF + `decoding_mode`，再安排上机重跑。
