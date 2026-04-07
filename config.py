import os
from dataclasses import dataclass

from dotenv import load_dotenv


# 在项目启动时优先加载 .env 文件中的配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)


@dataclass
class Config:
    """
    全局配置类

    说明：
    - 统一从环境变量中读取配置，便于在不同环境（本地 / 测试 / 生产）之间切换
    - .env 仅用于本地开发，生产环境建议使用系统环境变量注入
    """

    # Flask 相关
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    # SQLite 数据库 URL（历史字段，数仓与验证 Demo 已迁移至 Doris；保留以兼容旧配置）
    SQLITE_DB_URL: str = os.getenv("SQLITE_DB_URL", "sqlite:///./data_agent.db")

    # Apache Doris（MySQL 协议），Agent 查数、校验工具与 Text-to-SQL 的执行目标
    # 示例：mysql+pymysql://readonly:pass@127.0.0.1:9030/your_db
    DORIS_DB_URL: str = os.getenv("DORIS_DB_URL", "")

    # LLM（支持 OpenAI 及 OpenAI 兼容接口，如 DeepSeek）
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # API Base URL：为空则使用 OpenAI 默认地址；使用 DeepSeek 时填 https://api.deepseek.com/v1
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    # 模型名称：OpenAI 用 gpt-4o-mini；DeepSeek 用 deepseek-chat 或 deepseek-reasoner
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "deepseek-chat")
    # Embeddings 模型：OpenAI 用 text-embedding-3-small；DeepSeek 用 deepseek-embedding
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "deepseek-embedding"
    )

    # ChromaDB 本地持久化路径
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


config = Config()

