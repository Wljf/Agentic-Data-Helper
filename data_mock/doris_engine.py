"""Doris 连接（与 config.DORIS_DB_URL 一致，初始化脚本需写权限）。"""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import config


def get_doris_engine() -> Engine:
    if not config.DORIS_DB_URL:
        raise ValueError(
            "未配置 DORIS_DB_URL。请在 .env 中设置，例如："
            "mysql+pymysql://user:pass@127.0.0.1:9030/your_db"
        )
    return create_engine(
        config.DORIS_DB_URL,
        echo=False,
        future=True,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
