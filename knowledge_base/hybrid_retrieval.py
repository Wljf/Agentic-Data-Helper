"""
混合检索：向量召回（Chroma） + BM25 词法召回，RRF 融合后精排（CrossEncoder 或双塔余弦）。

运行前需执行：python -m knowledge_base.build_rag（生成分块向量库与 kb_manifest.json）。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import jieba
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

from config import config

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_manifest: Optional[List[Dict[str, Any]]] = None
_bm25: Optional[BM25Okapi] = None
_tokenized_corpus: Optional[List[List[str]]] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_chroma_store: Optional[Chroma] = None
_cross_encoder = None


def _manifest_path() -> str:
    return os.path.join(os.path.abspath(config.CHROMA_PERSIST_DIR), "kb_manifest.json")


def _load_manifest() -> List[Dict[str, Any]]:
    global _manifest
    if _manifest is not None:
        return _manifest
    path = _manifest_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到 {path}，请先执行：python -m knowledge_base.build_rag"
        )
    with open(path, "r", encoding="utf-8") as f:
        _manifest = json.load(f)
    return _manifest


def _tokenize(text: str) -> List[str]:
    """中英混合分词（BM25 用）。"""
    text = (text or "").strip().lower()
    if not text:
        return []
    return [t for t in jieba.lcut(text) if t.strip()]


def _ensure_bm25() -> Tuple[BM25Okapi, List[List[str]]]:
    global _bm25, _tokenized_corpus
    if _bm25 is not None and _tokenized_corpus is not None:
        return _bm25, _tokenized_corpus
    manifest = _load_manifest()
    texts = [item["text"] for item in manifest]
    _tokenized_corpus = [_tokenize(t) for t in texts]
    _bm25 = BM25Okapi(_tokenized_corpus)
    return _bm25, _tokenized_corpus


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_chroma() -> Chroma:
    global _chroma_store
    if _chroma_store is not None:
        return _chroma_store
    persist = os.path.abspath(config.CHROMA_PERSIST_DIR)
    if not os.path.isdir(persist):
        raise FileNotFoundError(
            f"Chroma 目录不存在：{persist}，请先执行 python -m knowledge_base.build_rag"
        )
    _chroma_store = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )
    return _chroma_store


def _rrf_fuse(
    ranked_ids: List[List[str]], k: int = 60
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    for ranks in ranked_ids:
        for rank, cid in enumerate(ranks):
            scores[cid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    model_name = os.getenv(
        "RAG_CROSS_ENCODER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    )
    from sentence_transformers import CrossEncoder

    _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


def _rerank_embedding(
    query: str, chunk_ids: List[str], texts: Dict[str, str], top_n: int
) -> List[str]:
    """用与向量库相同的句向量做二次打分（轻量精排）。"""
    emb = get_embeddings()
    qv = np.array(emb.embed_query(query))
    qn = np.linalg.norm(qv)
    scored: List[Tuple[str, float]] = []
    for cid in chunk_ids:
        t = texts.get(cid, "")
        if not t:
            continue
        dv = np.array(emb.embed_documents([t])[0])
        dn = np.linalg.norm(dv)
        if dn == 0 or qn == 0:
            sim = 0.0
        else:
            sim = float(np.dot(qv, dv) / (qn * dn))
        scored.append((cid, sim))
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:top_n]]


def _rerank_cross_encoder(
    query: str, chunk_ids: List[str], texts: Dict[str, str], top_n: int
) -> List[str]:
    try:
        ce = _get_cross_encoder()
    except Exception:
        return _rerank_embedding(query, chunk_ids, texts, top_n)
    valid_ids = [cid for cid in chunk_ids if cid in texts]
    if not valid_ids:
        return chunk_ids[:top_n]
    pairs = [(query, texts[cid]) for cid in valid_ids]
    scores = ce.predict(pairs)
    order = np.argsort(-np.array(scores))
    return [valid_ids[i] for i in order[:top_n]]


def hybrid_search(
    query: str,
    final_k: int = 5,
    vector_k: Optional[int] = None,
    bm25_k: Optional[int] = None,
    fusion_pool: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    混合检索 + 精排。

    流程：向量 Top-Kv + BM25 Top-Kb → RRF 融合 → 取 fusion_pool 条 → 精排 → final_k。
    """
    if not query or not query.strip():
        return []

    vector_k = vector_k or int(os.getenv("RAG_VECTOR_RECALL_K", "24"))
    bm25_k = bm25_k or int(os.getenv("RAG_BM25_RECALL_K", "24"))
    fusion_pool = fusion_pool or int(os.getenv("RAG_FUSION_POOL", "30"))

    manifest = _load_manifest()
    id_to_text = {item["chunk_id"]: item["text"] for item in manifest}
    id_order = [item["chunk_id"] for item in manifest]

    # --- 向量召回 ---
    store = get_chroma()
    vec_docs = store.similarity_search_with_score(
        query, k=min(vector_k, max(len(manifest), 1))
    )
    vec_ranked: List[str] = []
    seen = set()
    for doc, _score in vec_docs:
        cid = (doc.metadata or {}).get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            vec_ranked.append(cid)

    # --- BM25 ---
    bm25, _ = _ensure_bm25()
    q_tokens = _tokenize(query)
    if not q_tokens:
        bm25_scores = [0.0] * len(manifest)
    else:
        bm25_scores = bm25.get_scores(q_tokens)
    bm25_idx = np.argsort(-np.array(bm25_scores))[:bm25_k]
    bm25_ranked = [id_order[i] for i in bm25_idx if 0 <= i < len(id_order)]

    # --- RRF ---
    fused = _rrf_fuse([vec_ranked, bm25_ranked])
    pool_ids = [cid for cid, _ in fused[:fusion_pool]]

    # --- 精排：默认双塔余弦（与向量模型一致，无额外下载）；可设 RAG_RERANK_MODE=cross_encoder ---
    rerank_mode = os.getenv("RAG_RERANK_MODE", "embedding").lower()
    if rerank_mode == "cross_encoder":
        final_ids = _rerank_cross_encoder(query, pool_ids, id_to_text, final_k)
    else:
        final_ids = _rerank_embedding(query, pool_ids, id_to_text, final_k)

    out: List[Dict[str, Any]] = []
    mid = {item["chunk_id"]: item for item in manifest}
    for cid in final_ids:
        if cid in mid:
            meta = mid[cid].get("metadata", {})
            out.append(
                {
                    "chunk_id": cid,
                    "text": mid[cid]["text"],
                    "metadata": meta,
                }
            )
    return out[:final_k]
