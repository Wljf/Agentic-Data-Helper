"""
Agent 工具函数集合（基于 LangChain @tool）。

本模块提供以下工具：
1. check_pk_tool
   - 校验指定表的逻辑主键是否唯一且非空
2. check_volume_tool
   - 对比 t 日与 t-1 日两天的数据量，判断是否出现异常波动
3. query_table_metadata_tool
   - 查询 Doris 表结构信息（字段名、数据类型等，基于 information_schema）
4. rag_definition_tool
   - 业务口径混合检索：向量 + BM25（RRF 融合）+ 精排（双塔余弦或 CrossEncoder）
"""

import re
from typing import List

from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import config
from knowledge_base.hybrid_retrieval import get_chroma, hybrid_search


# --- 数据库与向量库基础设施 ----------------------------------------------------

_engine: Engine | None = None


def get_analytics_engine() -> Engine:
    """
    获取分析库 Engine（单例，Apache Doris，MySQL 协议）。

    说明：
    - 使用 config.DORIS_DB_URL（SQLAlchemy URL，例如 mysql+pymysql://...）
    - 生产环境请使用只读账号
    """
    global _engine
    if _engine is None:
        if not config.DORIS_DB_URL:
            raise ValueError(
                "未配置 DORIS_DB_URL，请在 .env 中设置 Doris 连接串，"
                "例如：mysql+pymysql://user:pass@127.0.0.1:9030/your_db"
            )
        _engine = create_engine(
            config.DORIS_DB_URL,
            echo=False,
            future=True,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


def get_sqlite_engine() -> Engine:
    """兼容旧名称：与 get_analytics_engine() 相同（当前分析库为 Doris）。"""
    return get_analytics_engine()


def get_chroma_vector_store():
    """
    获取 Chroma 向量库（单例，与 hybrid_retrieval 共用）。

    说明：
    - 依赖 knowledge_base/build_rag.py 预先构建持久化向量库与 kb_manifest.json
    """
    return get_chroma()


def _safe_identifier(name: str) -> str:
    """
    对表名 / 字段名做简单的白名单校验，避免 SQL 注入。

    规则：
    - 只能包含字母、数字和下划线
    - 必须以字母或下划线开头
    """
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"非法标识符：{name}")
    return name


# --- 工具 1：主键唯一性与非空检查 ----------------------------------------------


@tool
def check_pk_tool(table_name: str, pk_column: str) -> str:
    """
    校验指定表的逻辑主键是否唯一且非空。

    参数：
    - table_name: 表名，例如 "dwd_trade_order_di"
    - pk_column: 逻辑主键字段名，例如 "order_id"

    返回：
    - 一段 Markdown 文本，包含：
      - 总记录数
      - 主键非空记录数
      - 主键去重后的记录数
      - 重复主键条数
      - 主键为空的条数
    """
    table = _safe_identifier(table_name)
    pk = _safe_identifier(pk_column)

    engine = get_analytics_engine()

    sql = text(
        f"""
        SELECT
            COUNT(1) AS total_count,
            COUNT({pk}) AS non_null_count,
            COUNT(DISTINCT {pk}) AS distinct_non_null_count
        FROM {table}
        """
    )

    with engine.connect() as conn:
        row = conn.execute(sql).one()

    total_count = int(row.total_count)
    non_null_count = int(row.non_null_count)
    distinct_non_null_count = int(row.distinct_non_null_count)
    null_pk_count = total_count - non_null_count
    duplicate_pk_count = non_null_count - distinct_non_null_count

    status_lines: List[str] = []
    if null_pk_count == 0 and duplicate_pk_count == 0:
        status_lines.append("**结论：主键字段在当前表中唯一且非空。** ✅")
    else:
        status_lines.append("**结论：检测到主键字段存在异常，请重点关注。** ⚠️")

    detail = f"""
### 主键完整性检查结果（表：`{table}`，主键字段：`{pk}`）

- 总记录数：`{total_count}`
- 主键非空记录数：`{non_null_count}`
- 主键去重后的记录数：`{distinct_non_null_count}`
- 主键为空记录数：`{null_pk_count}`
- 主键重复记录数（非空且重复）：`{duplicate_pk_count}`

{chr(10).join(status_lines)}
"""
    return detail.strip()


# --- 工具 2：两日数据量波动检查 -------------------------------------------------


@tool
def check_volume_tool(table_name: str, dt_t: str, dt_t_minus_1: str) -> str:
    """
    对比同一张表在 t 日与 t-1 日的记录数，用于快速发现数据量异常波动。

    参数：
    - table_name: 表名，例如 "dwd_trade_order_di"
    - dt_t: t 日的分区值（字符串形式的日期，如 "2026-03-08"）
    - dt_t_minus_1: t-1 日的分区值

    返回：
    - 一段 Markdown 文本，包含两天的记录数和波动结论。
    - 如果 t 日数据量小于 t-1 日，将在文本中包含 "[WARNING]"。
    """
    table = _safe_identifier(table_name)

    engine = get_analytics_engine()

    with engine.connect() as conn:
        sql_t = text(f"SELECT COUNT(1) AS cnt FROM {table} WHERE dt = :dt")
        sql_t_minus_1 = text(f"SELECT COUNT(1) AS cnt FROM {table} WHERE dt = :dt")

        cnt_t = int(conn.execute(sql_t, {"dt": dt_t}).scalar() or 0)
        cnt_t_minus_1 = int(conn.execute(sql_t_minus_1, {"dt": dt_t_minus_1}).scalar() or 0)

    diff = cnt_t - cnt_t_minus_1
    ratio = (cnt_t / cnt_t_minus_1) if cnt_t_minus_1 > 0 else None

    warning_flag = cnt_t < cnt_t_minus_1
    flag_text = "[WARNING]" if warning_flag else "[OK]"

    ratio_str = f"{ratio:.2%}" if ratio is not None else "N/A"

    detail = f"""
### 数据量波动检查结果（表：`{table}`）

- t-1 日（`{dt_t_minus_1}`）记录数：`{cnt_t_minus_1}`
- t 日（`{dt_t}`）记录数：`{cnt_t}`
- 绝对变化量：`{diff}`
- 相对变化比例（t / t-1）：`{ratio_str}`

**结论：{flag_text}**

- 若为 [WARNING]，说明 t 日数据量低于 t-1 日，请排查上游任务是否延迟或失败。
- 若为 [OK]，说明两日数据量整体稳定或在可接受波动范围内。
"""
    return detail.strip()


# --- 工具 2.5：列出所有表名（供 Agentic RAG 先发现表再查元数据） -------------


@tool
def list_tables_tool() -> str:
    """
    列出当前 Doris 数据库（当前连接默认库）中所有 BASE TABLE 的表名。

    返回：
    - 一段 Markdown 文本，包含表名列表；若无表则提示为空。
    """
    engine = get_analytics_engine()
    sql = text(
        """
        SELECT TABLE_NAME AS name
        FROM information_schema.tables
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    names = [row.name for row in rows if row.name]
    if not names:
        return "当前数据库中没有任何用户表。"
    return "### 当前数据库中的表\n\n" + "\n".join(f"- `{n}`" for n in names)


# --- 工具 3：表结构元数据查询 ---------------------------------------------------


@tool
def query_table_metadata_tool(table_name: str) -> str:
    """
    查询 Doris 表结构信息（字段名、数据类型、是否允许为空、是否为主键列）。

    参数：
    - table_name: 表名，例如 "dwd_trade_order_di"

    返回：
    - 一段 Markdown 文本，包含字段级元数据信息。

    说明：
    - 使用 information_schema.COLUMNS（与 MySQL 兼容）。
    """
    table = _safe_identifier(table_name)
    engine = get_analytics_engine()

    meta_sql = text(
        """
        SELECT
            COLUMN_NAME AS col_name,
            DATA_TYPE AS data_type,
            IS_NULLABLE AS is_nullable,
            COLUMN_KEY AS column_key
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = :table_name
        ORDER BY ORDINAL_POSITION
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(meta_sql, {"table_name": table}).fetchall()

    if not rows:
        return f"未在当前 Doris 数据库中找到表 `{table}`，请确认表名是否正确。"

    header = "| 字段名 | 数据类型 | 是否非空 | 是否主键 |\n|--------|----------|----------|----------|"
    lines = [header]
    for row in rows:
        m = row._mapping
        name = m["col_name"]
        col_type = (m.get("data_type") or "") or ""
        is_nullable = (m.get("is_nullable") or "").upper()
        notnull = "YES" if is_nullable == "NO" else "NO"
        col_key = (m.get("column_key") or "").upper()
        is_pk = "YES" if col_key == "PRI" else "NO"
        lines.append(f"| `{name}` | `{col_type}` | {notnull} | {is_pk} |")

    md = f"""
### 表结构信息：`{table}`

> 数据源：information_schema.COLUMNS（TABLE_SCHEMA = DATABASE()）

{chr(10).join(lines)}
"""
    return md.strip()


# --- 工具 4：RAG 检索业务口径定义 ----------------------------------------------


@tool
def rag_definition_tool(query: str, top_k: int = 5) -> str:
    """
    基于混合检索（向量 + BM25 召回，RRF 融合，精排）从业务口径知识库取回相关片段。

    参数：
    - query: 用户自然语言问题，例如“GMV 是如何定义的？”
    - top_k: 最终返回的片段数量（默认 5）

    返回：
    - 一段 Markdown 文本，其中包含与 query 最相关的若干业务口径定义片段。
    """
    try:
        hits = hybrid_search(query, final_k=top_k)
    except Exception as e:
        return f"业务口径检索失败：{e}"

    if not hits:
        return "在业务口径知识库中未找到相关定义，请尝试换个描述方式。"

    parts = ["### 业务口径知识库检索结果（向量 + BM25 混合，精排）"]
    for idx, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        source = meta.get("source", "unknown")
        cid = h.get("chunk_id", "")
        parts.append(f"\n#### 命中片段 {idx}（`{source}` · `{cid}`）\n")
        parts.append((h.get("text") or "").strip())

    return "\n".join(parts).strip()


__all__ = [
    "check_pk_tool",
    "check_volume_tool",
    "list_tables_tool",
    "query_table_metadata_tool",
    "rag_definition_tool",
    "get_analytics_engine",
    "get_sqlite_engine",
]

