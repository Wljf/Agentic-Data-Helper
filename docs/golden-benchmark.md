# 评测指标说明（SQL + RAG 口径）

## 一、SQL 查询准确率（结果集一致）

在无「自然语言 → 唯一正确 SQL」的人工标注时，采用 **标准 SQL** 在 Doris 上执行得到 **真值结果集**，与 **模型生成的 SQL** 的执行结果对比。

### 指标

| 指标 | 含义 |
|------|------|
| **sql_query_result_accuracy**（主） | 各用例中，生成 SQL 与标准 SQL **在 Doris 上执行后的结果集**是否一致（列值规范化、行序无关）。即「查出来的数对不对」。 |
| **sql_string_match_accuracy**（辅） | 规范化后的 SQL **文本**是否与标准 SQL 完全一致。写法不同但结果相同时，主指标仍可为真。 |

历史字段别名：`summary.aliases.result_accuracy`、`summary.aliases.sql_string_accuracy`。

### 前置条件

1. 已配置 `DORIS_DB_URL`，且已执行 `python -m data_mock.init_warehouse` 导入与 `golden_sql.json` 一致的数据。
2. 已配置 `OPENAI_API_KEY`（Text-to-SQL 会调模型）。

### 运行

```bash
python -m evaluation.run_benchmark
python -m evaluation.run_benchmark --json evaluation/golden_sql.json --out data/benchmark_report.json
```

### 扩展用例

编辑 `evaluation/golden_sql.json`：`question`、`expected_sql`；默认 **`sql_only_eval: true`** 仅测 Text-to-SQL（稳定）。详见文件内说明。

---

## 二、RAG 口径问答准确率（有检索 vs 无检索）

针对**业务口径、指标定义**等自然语言问题：用验证集中的 **标准答案**（由知识库文档人工摘录）与模型生成答案对比。

### 指标

| 模式 | 含义 |
|------|------|
| **with_rag** | 先 `rag_definition_tool` 检索 Chroma，再将检索片段 + 问题交给 LLM 生成回答。 |
| **without_rag** | **不检索**，同一 LLM 在无内部知识上下文时直接回答（基线）。 |
| **mean_similarity** | 标准答案与生成答案的 **句向量余弦相似度**（与 `knowledge_base/build_rag` 使用同一 Embedding 模型）。 |
| **accuracy_percent** | 相似度 ≥ `similarity_threshold`（默认见 `rag_golden.json`）的样本占比。 |
| **rag_lift_accuracy_points** | `with_rag` 准确率 − `without_rag` 准确率（百分点），衡量检索带来的增益。 |

### 前置条件

1. 已执行 `python -m knowledge_base.build_rag` 构建 Chroma。
2. 已配置 `OPENAI_API_KEY`。

### 运行

```bash
python -m evaluation.run_rag_benchmark
python -m evaluation.run_rag_benchmark --json evaluation/rag_golden.json --out data/rag_benchmark_report.json
```

### 扩展验证集

编辑 `evaluation/rag_golden.json`：为每条补充 `question` 与 `ground_truth_answer`（建议与 `definitions.md` / `business_definitions.md` 一致）。

---

## 三、合并运行

```bash
python -m evaluation.run_all_metrics
```

会依次执行 SQL 评测与 RAG 评测，并在成功时输出 `headline` 汇总字段。
