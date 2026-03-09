"""
多层级电商数仓模拟数据初始化脚本（SQLite）。

生成 ODS / DWD / DWS / ADS 四层表结构与至少连续 3 天的数据，
模拟大促前后数据波动，供 Agentic RAG 查数找数使用。

表说明：
- ODS: ods_log_user_action_di  用户行为日志（按天分区）
- DWD: dwd_trade_order_detail_di 订单明细（order_id, user_id, sku_id, pay_amount, dt）
- DWS: dws_user_trade_summary_nd 用户粒度交易汇总（user_id, is_new_user, total_amount, order_count, dt）
- ADS: ads_sales_dashboard_di  大盘销售看板
"""

import random
from datetime import date, timedelta
from typing import List

import pandas as pd
from faker import Faker
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine

from config import config

fake = Faker("zh_CN")


def get_engine() -> Engine:
    if not config.SQLITE_DB_URL:
        raise ValueError("未配置 SQLITE_DB_URL")
    return create_engine(config.SQLITE_DB_URL, echo=False, future=True)


def run_ddl(engine: Engine, statements: List[str]) -> None:
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def create_ods_table(engine: Engine) -> None:
    """ODS 层：用户行为日志，按天分区（dt）。"""
    run_ddl(engine, ["""
    CREATE TABLE IF NOT EXISTS ods_log_user_action_di (
        log_id TEXT,
        user_id INTEGER NOT NULL,
        action_type TEXT NOT NULL,
        page TEXT,
        extra_json TEXT,
        dt TEXT NOT NULL
    );
    """])


def create_dwd_table(engine: Engine) -> None:
    """DWD 层：订单明细。"""
    run_ddl(engine, ["""
    CREATE TABLE IF NOT EXISTS dwd_trade_order_detail_di (
        order_id TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        sku_id TEXT NOT NULL,
        pay_amount REAL NOT NULL,
        quantity INTEGER NOT NULL,
        dt TEXT NOT NULL
    );
    """])


def create_dws_table(engine: Engine) -> None:
    """DWS 层：用户粒度交易汇总（含新老客标签）。"""
    run_ddl(engine, ["""
    CREATE TABLE IF NOT EXISTS dws_user_trade_summary_nd (
        user_id INTEGER NOT NULL,
        is_new_user INTEGER NOT NULL,
        total_amount REAL NOT NULL,
        order_count INTEGER NOT NULL,
        dt TEXT NOT NULL
    );
    """])


def create_ads_table(engine: Engine) -> None:
    """ADS 层：大盘销售看板。"""
    run_ddl(engine, ["""
    CREATE TABLE IF NOT EXISTS ads_sales_dashboard_di (
        dt TEXT NOT NULL,
        gmv REAL NOT NULL,
        order_count INTEGER NOT NULL,
        new_user_count INTEGER NOT NULL,
        old_user_count INTEGER NOT NULL,
        atv_new REAL,
        atv_old REAL
    );
    """])


def generate_ods_for_date(d: date, n: int) -> pd.DataFrame:
    action_types = ["click", "pv", "cart", "order", "pay"]
    pages = ["home", "list", "detail", "cart", "order"]
    rows = []
    for i in range(n):
        rows.append({
            "log_id": fake.uuid4(),
            "user_id": random.randint(1, 5000),
            "action_type": random.choice(action_types),
            "page": random.choice(pages),
            "extra_json": "{}",
            "dt": d.isoformat(),
        })
    return pd.DataFrame(rows)


def generate_dwd_for_date(d: date, n: int, promo_boost: float = 1.0) -> pd.DataFrame:
    """订单明细；promo_boost 模拟大促日单量与金额放大。"""
    rows = []
    for _ in range(n):
        order_id = fake.uuid4()
        user_id = random.randint(1, 5000)
        for _ in range(random.randint(1, 3)):
            pay_amount = round(random.uniform(20, 800) * promo_boost, 2)
            rows.append({
                "order_id": order_id,
                "user_id": user_id,
                "sku_id": fake.ean13(),
                "pay_amount": pay_amount,
                "quantity": random.randint(1, 5),
                "dt": d.isoformat(),
            })
    return pd.DataFrame(rows)


def generate_dws_from_dwd(engine: Engine, d: date) -> None:
    """从 DWD 聚合出 DWS 用户汇总；新客 = 该用户在全表中最小的 dt 等于当日。"""
    sql = """
    INSERT INTO dws_user_trade_summary_nd (user_id, is_new_user, total_amount, order_count, dt)
    SELECT
        a.user_id,
        CASE WHEN (SELECT MIN(dt) FROM dwd_trade_order_detail_di WHERE user_id = a.user_id) = :dt THEN 1 ELSE 0 END AS is_new_user,
        a.total_amount,
        a.order_count,
        :dt AS dt
    FROM (
        SELECT user_id, SUM(pay_amount) AS total_amount, COUNT(DISTINCT order_id) AS order_count
        FROM dwd_trade_order_detail_di WHERE dt = :dt GROUP BY user_id
    ) a
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {"dt": d.isoformat()})


def generate_ads_for_date(engine: Engine, d: date) -> None:
    """从 DWS 聚合出 ADS 看板（含 GMV、新老客数、新老客客单价）。"""
    sql = """
    INSERT INTO ads_sales_dashboard_di (dt, gmv, order_count, new_user_count, old_user_count, atv_new, atv_old)
    SELECT
        :dt AS dt,
        SUM(total_amount) AS gmv,
        SUM(order_count) AS order_count,
        SUM(CASE WHEN is_new_user = 1 THEN 1 ELSE 0 END) AS new_user_count,
        SUM(CASE WHEN is_new_user = 0 THEN 1 ELSE 0 END) AS old_user_count,
        CASE WHEN SUM(CASE WHEN is_new_user = 1 THEN 1 ELSE 0 END) > 0
             THEN SUM(CASE WHEN is_new_user = 1 THEN total_amount ELSE 0 END) * 1.0 / SUM(CASE WHEN is_new_user = 1 THEN 1 ELSE 0 END) ELSE NULL END AS atv_new,
        CASE WHEN SUM(CASE WHEN is_new_user = 0 THEN 1 ELSE 0 END) > 0
             THEN SUM(CASE WHEN is_new_user = 0 THEN total_amount ELSE 0 END) * 1.0 / SUM(CASE WHEN is_new_user = 0 THEN 1 ELSE 0 END) ELSE NULL END AS atv_old
    FROM dws_user_trade_summary_nd
    WHERE dt = :dt
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {"dt": d.isoformat()})


def main():
    engine = get_engine()
    create_ods_table(engine)
    create_dwd_table(engine)
    create_dws_table(engine)
    create_ads_table(engine)

    # 连续 3 天：T-2 基准，T-1 略涨，T 大促日明显放大
    base = date.today() - timedelta(days=2)
    days = [base + timedelta(days=i) for i in range(3)]
    # 大促日倍数
    boosts = [1.0, 1.1, 1.5]

    for d, boost in zip(days, boosts):
        n_ods = random.randint(800, 1200)
        n_orders = int(random.randint(400, 600) * boost)
        df_ods = generate_ods_for_date(d, n_ods)
        df_dwd = generate_dwd_for_date(d, n_orders, promo_boost=boost)
        df_ods.to_sql("ods_log_user_action_di", con=engine, if_exists="append", index=False, method="multi", chunksize=500)
        df_dwd.to_sql("dwd_trade_order_detail_di", con=engine, if_exists="append", index=False, method="multi", chunksize=500)
        generate_dws_from_dwd(engine, d)
        generate_ads_for_date(engine, d)

    print("数仓初始化完成。")
    print("表：ods_log_user_action_di, dwd_trade_order_detail_di, dws_user_trade_summary_nd, ads_sales_dashboard_di")
    print("日期：", [d.isoformat() for d in days])


if __name__ == "__main__":
    main()
