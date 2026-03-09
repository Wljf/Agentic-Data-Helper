"""
数据验证场景（Validation Agent）。

职责：
- 根据用户的自然语言输入，解析出：
  - 表名（默认使用 dwd_trade_order_di）
  - 主键字段名（默认使用 order_id）
  - t 日与 t-1 日的日期分区
- 调用下游工具：
  - check_pk_tool
  - check_volume_tool
- 汇总工具返回结果，生成一份结构化的 Markdown 质量报告。

实现思路（为了简单可靠，本 Demo 不使用复杂的 LangChain Agent Executor）：
1. 使用 LLM 做一次“参数抽取”，只输出 JSON 结构
2. 在 Python 侧解析 JSON 字符串
3. 调用工具函数获取检查结果
4. 再使用 LLM 将检查结果组织成可读性较好的中文报告
"""

import json
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import config
from agents.tools import check_pk_tool, check_volume_tool


@dataclass
class ValidationParams:
    """
    由 LLM 抽取出的数据验证参数。
    """

    table_name: str
    pk_column: str
    dt_t: str
    dt_t_minus_1: str


def _get_llm() -> ChatOpenAI:
    """
    创建用于参数抽取和报告生成的 LLM。

    说明：
    - 这里使用一个相对轻量的模型（可根据实际情况调整）
    - temperature 设为 0，保证输出稳定、偏向确定性
    """
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


def _extract_params(user_input: str) -> ValidationParams:
    """
    使用 LLM 从用户自然语言中抽取数据验证所需的参数。

    输出格式为 JSON，仅包含以下字段：
    - table_name: 表名，字符串，若用户未提及则默认 "dwd_trade_order_di"
    - pk_column: 主键字段名，若用户未提及则默认 "order_id"
    - dt_t: t 日日期（YYYY-MM-DD）
    - dt_t_minus_1: t-1 日日期（YYYY-MM-DD）
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是数据仓库领域的专家，负责从中文自然语言中抽取结构化参数，用于数据质量检查。\n"
                "只允许输出合法的 JSON 字符串，不要添加任何解释文字。\n"
                "如果用户没有明确指定表名，则使用默认表名：\"dwd_trade_order_di\"。\n"
                "如果用户没有明确指定主键字段名，则使用默认主键字段名：\"order_id\"。\n"
                "日期字段 dt_t 和 dt_t_minus_1 必须是 YYYY-MM-DD 形式的字符串，如果用户说“今天”和“昨天”，需要转化为具体日期。\n",
            ),
            (
                "human",
                "用户输入：{user_input}\n\n"
                "请根据上述规则，从用户输入中抽取参数，并严格输出如下 JSON 结构（不要多也不要少）：\n"
                '{{"table_name": "...", "pk_column": "...", "dt_t": "YYYY-MM-DD", "dt_t_minus_1": "YYYY-MM-DD"}}',
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"user_input": user_input})
    content = response.content.strip()

    # 这里假设 LLM 按要求返回了合法的 JSON 字符串
    data = json.loads(content)

    return ValidationParams(
        table_name=data.get("table_name", "dwd_trade_order_di"),
        pk_column=data.get("pk_column", "order_id"),
        dt_t=data["dt_t"],
        dt_t_minus_1=data["dt_t_minus_1"],
    )


def _build_final_report(
    user_input: str,
    params: ValidationParams,
    pk_result_md: str,
    volume_result_md: str,
) -> str:
    """
    使用 LLM 将工具返回的 Markdown 拼接成一份更友好的质量报告。
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名资深数据质量分析师，请根据工具返回的检查结果，"
                "为业务方生成一份简洁、可读性强的中文数据质量报告，使用 Markdown 格式输出。\n"
                "报告中需要包含：\n"
                "1. 本次检查的背景（简要复述用户问题）\n"
                "2. 主键完整性检查结论与关键指标\n"
                "3. 数据量波动检查结论与关键指标\n"
                "4. 如果有异常，请给出简单的排查建议。\n",
            ),
            (
                "human",
                "用户原始问题：\n"
                "{user_input}\n\n"
                "主键检查结果（Markdown）：\n"
                "{pk_result_md}\n\n"
                "数据量波动检查结果（Markdown）：\n"
                "{volume_result_md}\n\n"
                "请据此生成最终的质量报告。",
            ),
        ]
    )

    chain = prompt | llm
    result = chain.invoke(
        {
            "user_input": user_input,
            "pk_result_md": pk_result_md,
            "volume_result_md": volume_result_md,
        }
    )
    return result.content.strip()


def run_validation_agent(user_input: str) -> str:
    """
    对外暴露的统一入口函数（供 Flask 路由调用）。

    流程：
    1. 使用 LLM 抽取参数（表名、主键字段、t / t-1 日期）
    2. 调用 check_pk_tool 和 check_volume_tool 获取检查结果
    3. 再次调用 LLM 生成结构化质量报告（Markdown）
    """
    params = _extract_params(user_input)

    # 调用工具（这里直接通过 .invoke 使用工具对象）
    pk_result_md: str = check_pk_tool.invoke(
        {"table_name": params.table_name, "pk_column": params.pk_column}
    )
    volume_result_md: str = check_volume_tool.invoke(
        {
            "table_name": params.table_name,
            "dt_t": params.dt_t,
            "dt_t_minus_1": params.dt_t_minus_1,
        }
    )

    final_report = _build_final_report(
        user_input=user_input,
        params=params,
        pk_result_md=pk_result_md,
        volume_result_md=volume_result_md,
    )
    return final_report


__all__ = ["run_validation_agent"]

