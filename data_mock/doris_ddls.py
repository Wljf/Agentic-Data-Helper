"""
Apache Doris 建表 DDL（Duplicate 模型，适合 Demo 与重复导入）。

说明：
- 使用 mysql+pymysql 连接 FE 的查询端口（默认 9030）执行 DDL/DML。
- replication_num=1 适用于单机 / 单副本开发环境；生产请按集群副本数调整。
- 列顺序需与 DUPLICATE KEY 前缀一致（Doris 要求）。
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def _run_ddl(engine: Engine, statements: list[str]) -> None:
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


# ODS：用户行为日志
DDL_ODS = """
CREATE TABLE IF NOT EXISTS ods_log_user_action_di (
    log_id VARCHAR(128) NOT NULL,
    user_id BIGINT NOT NULL,
    action_type VARCHAR(64) NOT NULL,
    page VARCHAR(128),
    extra_json VARCHAR(2048),
    dt VARCHAR(32) NOT NULL
)
DUPLICATE KEY(log_id, user_id)
DISTRIBUTED BY HASH(user_id) BUCKETS 10
PROPERTIES (
    "replication_num" = "1"
);
"""

# DWD：订单明细（一行多 SKU 时 order_id 可重复出现）
# DUPLICATE KEY 列须为表字段前缀，故顺序为 order_id, sku_id, dt, ...
DDL_DWD = """
CREATE TABLE IF NOT EXISTS dwd_trade_order_detail_di (
    order_id VARCHAR(128) NOT NULL,
    sku_id VARCHAR(64) NOT NULL,
    dt VARCHAR(32) NOT NULL,
    user_id BIGINT NOT NULL,
    pay_amount DOUBLE NOT NULL,
    quantity INT NOT NULL
)
DUPLICATE KEY(order_id, sku_id, dt)
DISTRIBUTED BY HASH(user_id) BUCKETS 10
PROPERTIES (
    "replication_num" = "1"
);
"""

# DWS：用户粒度汇总
DDL_DWS = """
CREATE TABLE IF NOT EXISTS dws_user_trade_summary_nd (
    user_id BIGINT NOT NULL,
    dt VARCHAR(32) NOT NULL,
    is_new_user INT NOT NULL,
    total_amount DOUBLE NOT NULL,
    order_count INT NOT NULL
)
DUPLICATE KEY(user_id, dt)
DISTRIBUTED BY HASH(user_id) BUCKETS 10
PROPERTIES (
    "replication_num" = "1"
);
"""

# ADS：大盘指标
DDL_ADS = """
CREATE TABLE IF NOT EXISTS ads_sales_dashboard_di (
    dt VARCHAR(32) NOT NULL,
    gmv DOUBLE NOT NULL,
    order_count BIGINT NOT NULL,
    new_user_count BIGINT NOT NULL,
    old_user_count BIGINT NOT NULL,
    atv_new DOUBLE,
    atv_old DOUBLE
)
DUPLICATE KEY(dt)
DISTRIBUTED BY HASH(dt) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);
"""

# 数据验证 Demo：订单表（含可空 order_id、重复主键等异常场景）
# DUPLICATE KEY 使用 (dt, user_id) 作为前缀，避免 order_id 为空时无法入库
DDL_DWD_TRADE_ORDER_DI = """
CREATE TABLE IF NOT EXISTS dwd_trade_order_di (
    dt VARCHAR(32) NOT NULL,
    user_id BIGINT NOT NULL,
    order_id VARCHAR(128),
    amount DOUBLE NOT NULL,
    status VARCHAR(64) NOT NULL
)
DUPLICATE KEY(dt, user_id)
DISTRIBUTED BY HASH(user_id) BUCKETS 10
PROPERTIES (
    "replication_num" = "1"
);
"""


def create_warehouse_tables_doris(engine: Engine) -> None:
    """创建 ODS / DWD / DWS / ADS 四层表。"""
    _run_ddl(engine, [DDL_ODS, DDL_DWD, DDL_DWS, DDL_ADS])


def create_validation_order_table_doris(engine: Engine) -> None:
    """创建数据验证场景使用的 dwd_trade_order_di 表。"""
    _run_ddl(engine, [DDL_DWD_TRADE_ORDER_DI])


def truncate_warehouse_tables_doris(engine: Engine) -> None:
    """清空数仓 Demo 表（ADS -> DWS -> DWD -> ODS 顺序）。"""
    stmts = [
        "TRUNCATE TABLE ads_sales_dashboard_di;",
        "TRUNCATE TABLE dws_user_trade_summary_nd;",
        "TRUNCATE TABLE dwd_trade_order_detail_di;",
        "TRUNCATE TABLE ods_log_user_action_di;",
    ]
    _run_ddl(engine, stmts)


def truncate_validation_order_table_doris(engine: Engine) -> None:
    _run_ddl(engine, ["TRUNCATE TABLE dwd_trade_order_di;"])
