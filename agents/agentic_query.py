"""
Agentic RAG：分治拆解 -> 迭代反思 -> 全局聚合，用于复杂查数找数场景。

本模块使用 LangGraph 将原有流程实现为标准状态机图：

   START
      |
      v
  node_decompose ──────────────────────────────────────────┐
      |                                                    │
      v                                                    │
  node_refine <────────────────────────────────────────────┼───┐
      |                                                    │   │
      v                                                    │   │
  node_retrieve  <───────────────────────────────────┐     │   │
      |                                             │     │   │
      v                                             │     │   │
  node_reflect ──(can_answer)───────────────────────┼─────┘   │
      |                                             │         │
      |  (not can_answer & round < 3) ──────────────┘         │
      |                                                       │
      v (not can_answer & round >= 3 或 can_answer)            │
  node_finalize_sub ──(current_sub_idx < len)─────────────────┘
      |
      v (current_sub_idx >= len)
  node_aggregate
      |
      v
    END
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import config
from agents.tools import (
    list_tables_tool,
    query_table_metadata_tool,
    rag_definition_tool,
)


# =============================================================================
# 全局状态定义 (AgentState)
# =============================================================================


class AgentState(TypedDict, total=False):
    """
    图的全局状态，用于在各节点之间传递信息。

    字段说明：
    - original_question: 用户原问题
    - sub_questions: 拆解后的子问题列表
    - current_sub_idx: 当前正在处理第几个子问题（0-based）
    - current_retrieval_round: 当前子问题的检索轮数（0 表示尚未检索，1 表示已检索 1 轮）
    - current_collected_contexts: 当前子问题已收集到的上下文（RAG/元数据检索结果）
    - refined_question: 当前子问题经重写后的文本（供 retrieve/reflect 使用）
    - reflect_can_answer: node_reflect 判断当前上下文是否足以回答
    - reflect_subquery: node_reflect 给出的下一轮检索用 subquery
    - reflect_used_fallback: 是否因 3 轮仍未找到而使用兜底（需加免责声明）
    - sub_results: 已完成的所有子问题的结论，供 node_aggregate 使用
    - final_answer: 最终输出的 SQL 和口径解释
    """

    original_question: str
    sub_questions: List[str]
    current_sub_idx: int
    current_retrieval_round: int
    current_collected_contexts: List[str]
    refined_question: str
    reflect_can_answer: bool
    reflect_subquery: Optional[str]
    reflect_used_fallback: bool
    sub_results: List[Dict[str, Any]]
    final_answer: str


# =============================================================================
# 内部辅助函数（保留原有 prompt 模板与工具）
# =============================================================================


def _get_llm() -> ChatOpenAI:
    if not config.OPENAI_API_KEY:
        raise ValueError("未配置 OPENAI_API_KEY，请在 .env 中设置。")
    kwargs = {
        "model": config.OPENAI_MODEL,
        "temperature": 0,
        "openai_api_key": config.OPENAI_API_KEY,
    }
    if config.OPENAI_API_BASE:
        kwargs["base_url"] = config.OPENAI_API_BASE.rstrip("/")
    return ChatOpenAI(**kwargs)


def _strip_markdown_code_block(raw: str) -> str:
    """剥离 LLM 输出中可能包裹的 Markdown 代码块（如 ```json ... ```）。"""
    s = (raw or "").strip()
    match = re.match(r"^```(?:\w*)\s*\n?(.*?)\n?```\s*$", s, re.DOTALL)
    if match:
        return match.group(1).strip()
    return s


# LLM 单例，供各节点复用
_llm: Optional[ChatOpenAI] = None


def _llm_instance() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = _get_llm()
    return _llm


# =============================================================================
# 节点 1：node_decompose
# =============================================================================


def node_decompose(state: AgentState) -> Dict[str, Any]:
    """
    调用大模型拆解问题，初始化 sub_questions。

    流转：START -> node_decompose -> node_refine
    """
    question = state.get("original_question", "") or ""
    if not question.strip():
        return {
            "sub_questions": [question],
            "current_sub_idx": 0,
            "current_collected_contexts": [],
            "current_retrieval_round": 0,
            "sub_results": state.get("sub_results") or [],
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个数据仓库分析助手。用户会提出一个需要「查数、找数、写 SQL」的复杂问题。"
                "请把该问题拆解为若干个**独立的**子问题，便于先查业务口径、再查表结构、最后写 SQL。"
                "子问题应覆盖：1）业务口径（如新老客定义、客单价、GMV 等）；2）数据来源（哪张表、哪些字段）。"
                "只输出一个 JSON 数组，每个元素是一个子问题字符串，不要其他解释。",
            ),
            (
                "human",
                "用户问题：\n{question}\n\n请输出 JSON 数组，例如：[\"子问题1\", \"子问题2\"]",
            ),
        ]
    )
    chain = prompt | _llm_instance()
    resp = chain.invoke({"question": question})
    raw = _strip_markdown_code_block(resp.content.strip())
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        arr = [question]
    if not isinstance(arr, list):
        arr = [question]
    sub_questions = [str(q).strip() for q in arr if str(q).strip()]
    if not sub_questions:
        sub_questions = [question]

    return {
        "sub_questions": sub_questions,
        "current_sub_idx": 0,
        "current_collected_contexts": [],
        "current_retrieval_round": 0,
        "sub_results": state.get("sub_results") or [],
    }


# =============================================================================
# 节点 2：node_refine
# =============================================================================


def node_refine(state: AgentState) -> Dict[str, Any]:
    """
    取出当前子问题，结合 sub_results 中之前的结论进行重写消歧。

    流转：node_decompose / node_finalize_sub -> node_refine -> node_retrieve

    注意：每次进入 refine 处理新子问题时，重置 current_collected_contexts 和 current_retrieval_round。
    """
    sub_questions = state.get("sub_questions") or []
    current_sub_idx = state.get("current_sub_idx", 0)
    sub_results = state.get("sub_results") or []

    if current_sub_idx >= len(sub_questions):
        # 理论上不应发生，兜底
        return {"refined_question": ""}

    sub_question = sub_questions[current_sub_idx]

    # 无已有结论时，无需重写
    if not sub_results:
        return {
            "refined_question": sub_question,
            "current_collected_contexts": [],
            "current_retrieval_round": 0,
        }

    prior_text = "\n".join(
        f"- 子问题：{a.get('sub_question', '')}\n  结论：{a.get('answer', '')}"
        for a in sub_results
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个数据助手。当前要回答的子问题可能依赖前面已经查到的「业务口径」或「表名/字段」。\n"
                "请根据「已有结论」把当前子问题重写为一句**自包含、无指代歧义**的新问题，便于单独做 RAG 或元数据检索。"
                "若无需修改，直接输出原问题。只输出重写后的问题，不要解释。",
            ),
            (
                "human",
                "已有结论：\n{prior_text}\n\n当前子问题：{sub_question}\n\n请输出重写后的子问题（一句话）：",
            ),
        ]
    )
    chain = prompt | _llm_instance()
    resp = chain.invoke({"prior_text": prior_text, "sub_question": sub_question})
    out = (resp.content or "").strip().strip('"')
    refined = out if out else sub_question

    return {
        "refined_question": refined,
        "current_collected_contexts": [],
        "current_retrieval_round": 0,
    }


# =============================================================================
# 节点 3：node_retrieve
# =============================================================================


def node_retrieve(state: AgentState) -> Dict[str, Any]:
    """
    根据重写后的子问题（或上一轮 reflect 提供的 subquery）查询 ChromaDB 口径或 Doris 表结构，
    将结果追加到 current_collected_contexts 中，并递增 current_retrieval_round。

    流转：node_refine -> node_retrieve -> node_reflect
    或：  node_reflect（can_answer=False & round<3）-> node_retrieve -> node_reflect
    """
    refined = state.get("refined_question", "")
    round_num = state.get("current_retrieval_round", 0)
    collected = list(state.get("current_collected_contexts") or [])

    if round_num == 0:
        # 第一轮：口径 RAG + 表列表
        def_text = rag_definition_tool.invoke({"query": refined, "top_k": 3})
        table_list_text = list_tables_tool.invoke({})
        retrieved = (
            f"【业务口径检索】\n{def_text}\n\n【当前数据库表列表】\n{table_list_text}"
        )
    else:
        # 后续轮：使用 reflect 提供的 subquery
        subquery = (state.get("reflect_subquery") or refined).strip()
        if not subquery:
            subquery = refined
        table_list_raw = list_tables_tool.invoke({})
        tables_found = re.findall(r"`(\w+)`", table_list_raw)
        chosen_table = None
        for t in tables_found:
            if t.lower() in subquery.lower() or subquery.lower() in t.lower():
                chosen_table = t
                break
        if chosen_table:
            retrieved = query_table_metadata_tool.invoke({"table_name": chosen_table})
        else:
            retrieved = rag_definition_tool.invoke({"query": subquery, "top_k": 3})
        retrieved = f"【第 {round_num + 1} 轮补充检索】\n{retrieved}"

    collected.append(retrieved)
    return {
        "current_collected_contexts": collected,
        "current_retrieval_round": round_num + 1,
    }


# =============================================================================
# 节点 4：node_reflect
# =============================================================================


def node_reflect(state: AgentState) -> Dict[str, Any]:
    """
    调用大模型判断 current_collected_contexts 是否足以回答当前子问题。
    设置 reflect_can_answer、reflect_subquery、reflect_used_fallback。

    流转：node_retrieve -> node_reflect
    之后由 route_after_reflect 决定：finalize_sub / 回到 retrieve
    """
    refined = state.get("refined_question", "")
    collected = state.get("current_collected_contexts") or []
    round_num = state.get("current_retrieval_round", 0)
    max_rounds = 3

    retrieved = "\n".join(collected) if collected else ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个数据仓库分析助手。根据「当前子问题」和「检索到的内容」，判断是否已经能完整回答该子问题。\n"
                "若检索内容中已包含所需的口径定义或表结构信息，则 can_answer 为 true。\n"
                "若缺少关键信息（例如没有提到相关表、或没有新老客/客单价等口径），则 can_answer 为 false，"
                "并给出下一轮检索用的 subquery（一句简短的关键词或问题，用于查口径或查表结构）。\n"
                "只输出一个 JSON 对象：{{\"can_answer\": true/false, \"reason\": \"简短原因\", \"subquery\": \"下一轮检索用的问题或关键词，若无则 null}}",
            ),
            (
                "human",
                "子问题：{sub_question}\n\n检索到的内容：\n{retrieved}\n\n当前轮次：{round_index}\n\n请输出 JSON：",
            ),
        ]
    )
    chain = prompt | _llm_instance()
    resp = chain.invoke({
        "sub_question": refined,
        "retrieved": retrieved,
        "round_index": round_num,
    })
    raw = _strip_markdown_code_block(resp.content.strip())
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = {"can_answer": False, "reason": "解析失败", "subquery": refined}

    can_answer = bool(obj.get("can_answer"))
    subquery = obj.get("subquery") if obj.get("subquery") else None
    # 达到最大轮数仍未找到时，强制走 finalize，并打上兜底标记
    used_fallback = not can_answer and round_num >= max_rounds

    return {
        "reflect_can_answer": can_answer,
        "reflect_subquery": subquery,
        "reflect_used_fallback": used_fallback,
    }


# =============================================================================
# 条件路由：route_after_reflect
# =============================================================================


def route_after_reflect(
    state: AgentState,
) -> Literal["finalize_sub", "retrieve"]:
    """
    在 node_reflect 之后的条件路由：

    - can_answer == True：走向 node_finalize_sub
    - can_answer == False 且 current_retrieval_round < 3：回到 node_retrieve 继续检索
    - can_answer == False 且 current_retrieval_round >= 3：强制走向 node_finalize_sub
      （node_finalize_sub 内部根据 reflect_used_fallback 打上兜底免责声明）
    """
    can_answer = state.get("reflect_can_answer", False)
    round_num = state.get("current_retrieval_round", 0)
    max_rounds = 3

    if can_answer:
        return "finalize_sub"
    if round_num < max_rounds:
        return "retrieve"
    return "finalize_sub"


# =============================================================================
# 节点 5：node_finalize_sub
# =============================================================================


def node_finalize_sub(state: AgentState) -> Dict[str, Any]:
    """
    将当前子问题收集到的 context 汇总，生成结论并存入 sub_results，
    同时 current_sub_idx += 1，重置 current_retrieval_round。
    若 reflect_used_fallback 为 True，则在结论中强制加入免责声明。

    流转：node_reflect -> node_finalize_sub
    之后由 route_after_finalize_sub 决定：回到 node_refine（下一题）/ node_aggregate
    """
    sub_questions = state.get("sub_questions") or []
    current_sub_idx = state.get("current_sub_idx", 0)
    collected = state.get("current_collected_contexts") or []
    sub_results = list(state.get("sub_results") or [])
    used_fallback = state.get("reflect_used_fallback", False)

    sub_question = sub_questions[current_sub_idx] if current_sub_idx < len(sub_questions) else ""
    answer = "\n\n".join(collected) if collected else ""

    if used_fallback:
        disclaimer = (
            "**免责声明**：未在当前数据库找到完全匹配的表结构，以下基于通用电商数仓经验推导。\n\n"
        )
        answer = disclaimer + answer

    sub_results.append({
        "sub_question": sub_question,
        "refined": state.get("refined_question", ""),
        "answer": answer,
        "used_fallback": used_fallback,
    })

    return {
        "sub_results": sub_results,
        "current_sub_idx": current_sub_idx + 1,
        "current_retrieval_round": 0,
        "current_collected_contexts": [],
    }


# =============================================================================
# 条件路由：route_after_finalize_sub
# =============================================================================


def route_after_finalize_sub(
    state: AgentState,
) -> Literal["refine", "aggregate"]:
    """
    在 node_finalize_sub 之后的条件路由：

    - current_sub_idx < len(sub_questions)：回到 node_refine 处理下一题
    - 否则：走向 node_aggregate 生成最终结果
    """
    sub_questions = state.get("sub_questions") or []
    current_sub_idx = state.get("current_sub_idx", 0)
    if current_sub_idx < len(sub_questions):
        return "refine"
    return "aggregate"


# =============================================================================
# 节点 6：node_aggregate
# =============================================================================


def node_aggregate(state: AgentState) -> Dict[str, Any]:
    """
    将所有 sub_results 喂给大模型，生成最终的只读 SELECT SQL 及口径通俗解释。

    流转：node_finalize_sub -> node_aggregate -> END
    """
    question = state.get("original_question", "")
    sub_results = state.get("sub_results") or []

    context_parts = []
    for r in sub_results:
        context_parts.append(
            f"### 子问题：{r.get('sub_question', '')}\n{r.get('answer', '')}"
        )
    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个 BI 分析助手，当前数据库为只读 Apache Doris（MySQL 协议）。\n"
                "请根据「用户原始问题」和「已集齐的子问答结论（业务口径 + 表结构信息）」完成两件事：\n"
                "1）生成一条**只读 SELECT** 查询 SQL，直接满足用户的查数需求；\n"
                "2）用通俗语言解释统计口径（如新老客、客单价、GMV 等）。\n"
                "安全要求：仅输出 SELECT，禁止 INSERT/UPDATE/DELETE/DROP 等任何写操作；不要多条 SQL。\n"
                "输出格式：先写「口径说明」段落，再写「SQL」段落，SQL 用 ```sql ... ``` 包裹。",
            ),
            (
                "human",
                "用户原始问题：\n{question}\n\n已集齐的上下文（子问答结论）：\n{context}\n\n"
                "请输出：1）口径说明；2）只读 SELECT SQL（用 ```sql 包裹）。",
            ),
        ]
    )
    chain = prompt | _llm_instance()
    resp = chain.invoke({"question": question, "context": context})
    final_answer = (resp.content or "").strip()

    return {"final_answer": final_answer}


# =============================================================================
# 构建 LangGraph 图
# =============================================================================


def _build_agentic_graph() -> StateGraph:
    """
    构建 Agentic RAG 状态机图。

    图结构：
        START -> decompose -> refine -> retrieve -> reflect
        reflect --[can_answer]-----------------------> finalize_sub
        reflect --[not can_answer & round<3]---------> retrieve (循环)
        reflect --[not can_answer & round>=3]--------> finalize_sub
        finalize_sub --[还有下一题]------------------> refine (循环)
        finalize_sub --[无下一题]---------------------> aggregate -> END
    """
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("decompose", node_decompose)
    builder.add_node("refine", node_refine)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("reflect", node_reflect)
    builder.add_node("finalize_sub", node_finalize_sub)
    builder.add_node("aggregate", node_aggregate)

    # 边：START -> decompose
    builder.add_edge(START, "decompose")

    # 边：decompose -> refine
    builder.add_edge("decompose", "refine")

    # 边：refine -> retrieve
    builder.add_edge("refine", "retrieve")

    # 边：retrieve -> reflect
    builder.add_edge("retrieve", "reflect")

    # 条件边：reflect 之后
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"finalize_sub": "finalize_sub", "retrieve": "retrieve"},
    )

    # 条件边：finalize_sub 之后
    builder.add_conditional_edges(
        "finalize_sub",
        route_after_finalize_sub,
        {"refine": "refine", "aggregate": "aggregate"},
    )

    # 边：aggregate -> END
    builder.add_edge("aggregate", END)

    return builder


# 编译后的图（单例）
_compiled_graph = None


def _get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_agentic_graph().compile()
    return _compiled_graph


# =============================================================================
# 对外统一入口
# =============================================================================


def run_agentic_rag(question: str, max_rounds: int = 3) -> str:
    """
    对外统一入口：执行 Agentic RAG 图并返回最终回答（含 SQL 与口径解释）。

    内部调用 app.invoke({"original_question": question, ...}) 运行整个图。
    max_rounds 为反思迭代检索的最大轮数（图中固定为 3，此处保留参数以兼容既有调用）。
    """
    app = _get_compiled_graph()
    initial: AgentState = {
        "original_question": question,
        "sub_questions": [],
        "current_sub_idx": 0,
        "current_retrieval_round": 0,
        "current_collected_contexts": [],
        "sub_results": [],
    }
    result = app.invoke(initial)
    return result.get("final_answer", "")


# 保留类形式以兼容可能的直接引用
class AgenticDataRAG:
    """基于 LangGraph 的 Agentic RAG 查数助手（委托给 run_agentic_rag）。"""

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds

    def run(self, question: str) -> str:
        return run_agentic_rag(question, max_rounds=self.max_rounds)


__all__ = ["AgenticDataRAG", "run_agentic_rag", "AgentState"]
