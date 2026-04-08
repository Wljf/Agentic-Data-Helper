"""评测脚本共用的 LLM 构造（避免与 agents 循环依赖）。"""

from langchain_openai import ChatOpenAI

from config import config


def get_eval_llm() -> ChatOpenAI:
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
