"""
基于 HuggingFace 句向量（与 knowledge_base/build_rag 及 agents/tools 中 Chroma 使用同一模型）
计算两段文本的余弦相似度，用于口径问答「生成答案 vs 标准答案」的自动打分。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

_EMB: Optional[HuggingFaceEmbeddings] = None

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    global _EMB
    if _EMB is None:
        _EMB = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _EMB


def cosine_similarity_texts(a: str, b: str) -> float:
    """返回 [0,1] 区间的余弦相似度（向量已 L2 归一化时等价于点积）。"""
    if not (a or "").strip() or not (b or "").strip():
        return 0.0
    emb = get_embeddings()
    va = np.array(emb.embed_query(a.strip()))
    vb = np.array(emb.embed_query(b.strip()))
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def accuracy_at_threshold(similarities: List[float], threshold: float) -> float:
    """准确率 = 相似度 >= 阈值 的样本比例。"""
    if not similarities:
        return 0.0
    ok = sum(1 for s in similarities if s >= threshold)
    return round(100.0 * ok / len(similarities), 2)
