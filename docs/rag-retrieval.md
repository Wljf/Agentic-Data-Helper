# 业务口径 RAG：分块 + 混合检索 + 精排

## 流程概览

1. **索引构建**（`python -m knowledge_base.build_rag`）  
   - 使用 `RecursiveCharacterTextSplitter` 对 `definitions.md`、`business_definitions.md` 分块。  
   - 每块写入 Chroma（向量），并写入 `CHROMA_PERSIST_DIR/kb_manifest.json`（文本与 `chunk_id`）。

2. **在线检索**（`knowledge_base/hybrid_retrieval.py`）  
   - **向量召回**：Chroma 相似度检索 Top-Kv（默认 24）。  
   - **BM25 召回**：对 manifest 全文建 `rank_bm25.BM25Okapi`，查询经 **jieba** 分词后取 Top-Kb（默认 24）。  
   - **RRF 融合**：对两路结果做 Reciprocal Rank Fusion，取融合池（默认 30 条）。  
   - **精排**  
     - 默认 `RAG_RERANK_MODE=embedding`：用与索引相同的句向量对「查询 vs 片段」做余弦相似度（轻量、无额外模型）。  
     - 可选 `RAG_RERANK_MODE=cross_encoder`：使用 `sentence_transformers.CrossEncoder`（默认 `cross-encoder/ms-marco-MiniLM-L-12-v2`，首次会下载权重）。

## 环境变量（可选）

| 变量 | 含义 | 默认 |
|------|------|------|
| `RAG_CHUNK_SIZE` | 分块字符数 | 520 |
| `RAG_CHUNK_OVERLAP` | 分块重叠 | 100 |
| `RAG_VECTOR_RECALL_K` | 向量召回条数 | 24 |
| `RAG_BM25_RECALL_K` | BM25 召回条数 | 24 |
| `RAG_FUSION_POOL` | RRF 后进入精排的候选数 | 30 |
| `RAG_RERANK_MODE` | `embedding` 或 `cross_encoder` | embedding |
| `RAG_CROSS_ENCODER_MODEL` | CrossEncoder 模型名 | cross-encoder/ms-marco-MiniLM-L-12-v2 |

## 重建索引后

请 **重启应用进程**，以便重新加载 `kb_manifest.json` 与 BM25 内存索引。
