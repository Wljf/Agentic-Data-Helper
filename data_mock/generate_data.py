"""
使用 Faker + Pandas 生成模拟订单明细数据，并写入本地 SQLite 数据库。

目标表：dwd_trade_order_di
字段：
- order_id: 主键，字符串
- user_id: 用户 ID，整数
- amount: 订单金额，浮点数
- status: 订单状态，字符串
- dt: 日期分区，格式为 YYYY-MM-DD（用于区分 t 与 t-1）

数据设计：
- 生成两天的数据：t-1（昨天）和 t（今天）
- t-1 日作为基准日，假设有 N 条数据
- t 日数据量 = N 的 80%（即少 20%），并在 t 日数据中故意制造以下异常：
  1) 若干条 order_id 重复
  2) 若干条 order_id 为空（None）

注意：
- 该脚本仅用于本地开发与演示，依赖 config.py 中的 SQLITE_DB_URL（一个本地 .db 文件）。
"""

import random
from datetime import date, timedelta

import pandas as pd
from faker import Faker
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine

from config import config


fake = Faker("zh_CN")


def get_write_engine() -> Engine:
    """
    获取用于写入的 SQLAlchemy Engine。

    说明：
    - 依赖 .env 中的 SQLITE_DB_URL
    - SQLite 为本地文件数据库，适合本地开发与 Demo 使用
    """
    if not config.SQLITE_DB_URL:
        raise ValueError("环境变量 SQLITE_DB_URL 未配置，请在 .env 中正确填写 SQLite 连接 URL。")
    return create_engine(config.SQLITE_DB_URL, echo=False, future=True)


def create_table_if_not_exists(engine: Engine) -> None:
    """
    若目标表不存在，则在 SQLite 中创建 dwd_trade_order_di 表。

    这里有一个重要设计点：
    - 为了让“主键校验工具”能够发现重复和空值，我们**不在数据库层强制主键约束**
    - 即逻辑主键为 order_id，但表结构中不设置 PRIMARY KEY 约束
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS dwd_trade_order_di (
        order_id TEXT,
        user_id INTEGER NOT NULL,
        amount REAL NOT NULL,
        status TEXT NOT NULL,
        dt TEXT NOT NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def generate_orders_for_date(target_date: date, num_rows: int) -> pd.DataFrame:
    """
    为指定日期生成 num_rows 条正常订单数据（不含刻意异常）。
    """
    records = []
    for _ in range(num_rows):
        order_id = fake.uuid4()
        user_id = random.randint(1, 10000)
        amount = round(random.uniform(10, 1000), 2)
        status = random.choice(["PAID", "UNPAID", "CANCELLED"])
        records.append(
            {
                "order_id": order_id,
                "user_id": user_id,
                "amount": amount,
                "status": status,
                "dt": target_date,
            }
        )
    return pd.DataFrame.from_records(records)


def inject_anomalies_for_today(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """
    在 t 日（今天）的数据中注入一些异常记录：
    - 部分 order_id 重复
    - 部分 order_id 为空

    说明：
    - 这里额外插入 5 条异常记录（3 条重复主键 + 2 条主键为空）
    - 为了保证“t 日总量比 t-1 日少约 20%”，主流程会在生成基础数据量时预留出这 5 条的空间
    """
    if df.empty:
        return df

    anomaly_records = []

    # 1) 制造 3 条 order_id 重复的数据（复制前 3 行的主键）
    duplicated_order_ids = df["order_id"].head(3).tolist()
    for oid in duplicated_order_ids:
        anomaly_records.append(
            {
                "order_id": oid,  # 与已有记录重复
                "user_id": random.randint(1, 10000),
                "amount": round(random.uniform(10, 1000), 2),
                "status": random.choice(["PAID", "UNPAID", "CANCELLED"]),
                "dt": target_date,
            }
        )

    # 2) 制造 2 条 order_id 为空的数据
    for _ in range(2):
        anomaly_records.append(
            {
                "order_id": None,  # 主键为空
                "user_id": random.randint(1, 10000),
                "amount": round(random.uniform(10, 1000), 2),
                "status": random.choice(["PAID", "UNPAID", "CANCELLED"]),
                "dt": target_date,
            }
        )

    anomaly_df = pd.DataFrame.from_records(anomaly_records)
    return pd.concat([df, anomaly_df], ignore_index=True)


def main():
    """
    主入口：
    - 连接 SQLite
    - 创建目标表（如不存在）
    - 生成 t-1 与 t 的数据并写入
    """
    engine = get_write_engine()
    create_table_if_not_exists(engine)

    # 定义两天：t-1（昨天）、t（今天）
    today = date.today()
    yesterday = today - timedelta(days=1)

    # 假设 t-1 有 N 条数据，t 有 N * 80% 条数据（再加上 5 条异常）
    base_count_yesterday = 1000
    anomaly_count_today = 5
    base_count_today = int(base_count_yesterday * 0.8) - anomaly_count_today

    if base_count_today <= 0:
        raise ValueError("基础数据量配置错误，请检查 base_count_yesterday 与 anomaly_count_today。")

    # 生成 t-1 日数据（全部为正常数据）
    df_yesterday = generate_orders_for_date(yesterday, base_count_yesterday)

    # 生成 t 日基础数据（正常数据）
    df_today_base = generate_orders_for_date(today, base_count_today)
    # 注入异常记录
    df_today = inject_anomalies_for_today(df_today_base, today)

    # 将数据写入 SQLite
    with engine.begin() as conn:
        df_yesterday.to_sql(
            "dwd_trade_order_di",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
        df_today.to_sql(
            "dwd_trade_order_di",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

    print("=== 数据生成完成 ===")
    print(f"日期 {yesterday} 插入记录数：{len(df_yesterday)}")
    print(f"日期 {today} 插入记录数：{len(df_today)}（含 {anomaly_count_today} 条刻意制造的异常）")
    print("目标表：dwd_trade_order_di")


if __name__ == "__main__":
    main()

