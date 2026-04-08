"""
RAG 口径问答评测：

1) 开启检索：Chroma 检索 + LLM 根据检索片段生成回答，与标准答案比相似度。
2) 关闭检索：同一 LLM 在无知识库上下文下直接回答，与标准答案比相似度。

指标：
- mean_similarity：各题余弦相似度均值
- accuracy_at_threshold：相似度 >= 阈值 的题目占比（默认阈值见 rag_golden.json）

用法：
  python -m evaluation.run_rag_benchmark
  python -m evaluation.run_rag_benchmark --json evaluation/rag_golden.json --out data/rag_benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from agents.tools import rag_definition_tool
from evaluation.llm_utils import get_eval_llm
from evaluation.text_similarity import accuracy_at_threshold, cosine_similarity_texts

PROMPT_WITH_RAG = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业数据口径助手。请**只根据下方「参考资料」**回答问题，表述简洁准确。"
            "若参考资料不足以回答，请明确说明资料中未提及，不要编造。",
        ),
        (
            "human",
            "参考资料：\n{context}\n\n问题：{question}\n\n请用 2～5 句中文回答。",
        ),
    ]
)

PROMPT_NO_RAG = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业数据口径助手。**当前未提供任何企业内部知识库或检索内容**。"
            "请仅依据常识作答；若无法从常识确定企业内部指标定义，请直接说明「无法在不访问业务知识库的情况下准确回答该企业内部口径」。",
        ),
        (
            "human",
            "问题：{question}\n\n请用 2～5 句中文回答。",
        ),
    ]
)


def answer_with_rag(question: str, top_k: int = 5) -> str:
    """检索知识库后由 LLM 生成回答。"""
    ctx = rag_definition_tool.invoke({"query": question, "top_k": top_k})
    if not ctx or "未找到" in ctx:
        ctx = "（检索无命中片段）"
    chain = PROMPT_WITH_RAG | get_eval_llm()
    resp = chain.invoke({"context": ctx, "question": question})
    return (resp.content or "").strip()


def answer_without_rag(question: str) -> str:
    """不检索，仅 LLM 直接回答。"""
    chain = PROMPT_NO_RAG | get_eval_llm()
    resp = chain.invoke({"question": question})
    return (resp.content or "").strip()


def load_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_mode(
    cases: List[Dict[str, Any]],
    use_rag: bool,
    threshold: float,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    sims: List[float] = []
    for c in cases:
        q = c["question"]
        gt = c["ground_truth_answer"]
        cid = c.get("id", "")
        try:
            if use_rag:
                gen = answer_with_rag(q)
            else:
                gen = answer_without_rag(q)
            sim = cosine_similarity_texts(gt, gen)
            sims.append(sim)
            details.append(
                {
                    "id": cid,
                    "question": q,
                    "ground_truth_answer": gt,
                    "generated_answer": gen,
                    "similarity": round(sim, 4),
                    "pass": sim >= threshold,
                }
            )
        except Exception as e:
            details.append(
                {
                    "id": cid,
                    "question": q,
                    "error": str(e),
                    "pass": False,
                }
            )
            sims.append(0.0)

    mean_sim = sum(sims) / len(sims) if sims else 0.0
    return {
        "mode": "with_rag" if use_rag else "without_rag",
        "threshold": threshold,
        "case_count": len(cases),
        "mean_similarity": round(mean_sim, 4),
        "accuracy_percent": accuracy_at_threshold(sims, threshold),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 口径问答评测")
    parser.add_argument(
        "--json",
        default=os.path.join(os.path.dirname(__file__), "rag_golden.json"),
        help="验证集 JSON",
    )
    parser.add_argument("--out", default="", help="输出报告路径")
    args = parser.parse_args()

    bundle = load_bundle(args.json)
    cases = bundle.get("cases", [])
    threshold = float(bundle.get("similarity_threshold", 0.72))

    report = {
        "source": args.json,
        "similarity_threshold": threshold,
        "note": "相似度为句向量余弦；准确率指 similarity>=阈值 的占比。无检索基线用于对比 RAG 增益。",
        "with_rag": run_mode(cases, use_rag=True, threshold=threshold),
        "without_rag": run_mode(cases, use_rag=False, threshold=threshold),
    }

    # RAG 增益（百分点）：有检索准确率 - 无检索准确率
    wr = report["with_rag"]["accuracy_percent"]
    wnr = report["without_rag"]["accuracy_percent"]
    report["rag_lift_accuracy_points"] = round(wr - wnr, 2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n已写入：{args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
