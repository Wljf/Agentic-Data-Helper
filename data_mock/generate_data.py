"""
使用 Faker + Pandas 生成模拟订单明细数据，并写入 Apache Doris。

目标表：dwd_trade_order_di
字段：
- order_id: 逻辑主键，字符串（可为空，用于演示异常）
- user_id: 用户 ID，整数
- amount: 订单金额，浮点数
- status: 订单状态，字符串
- dt: 日期分区，格式为 YYYY-MM-DD（用于区分 t 与 t-1）

数据设计：
- 生成两天的数据：t-1（昨天）和 t（今天）
- t 日数据中故意制造：重复 order_id、order_id 为空

依赖：
- .env 中 DORIS_DB_URL（需具备建表与写入权限）

环境变量：
- INIT_VALIDATION_TRUNCATE=true  导入前清空 dwd_trade_order_di
"""

import os
import random
from datetime import date, timedelta

import pandas as pd
from faker import Faker

from .doris_ddls import create_validation_order_table_doris, truncate_validation_order_table_doris
from .doris_engine import get_doris_engine

fake = Faker("zh_CN")


def get_write_engine():
    return get_doris_engine()


def create_table_if_not_exists(engine) -> None:
    """
    若目标表不存在，则在 Doris 中创建 dwd_trade_order_di。

    逻辑主键为 order_id，但表结构中不强制 UNIQUE，便于演示重复与空值。
    """
    create_validation_order_table_doris(engine)


def generate_orders_for_date(target_date: date, num_rows: int) -> pd.DataFrame:
    """为指定日期生成 num_rows 条正常订单数据（不含刻意异常）。"""
    records = []
    dt_str = target_date.isoformat()
    for _ in range(num_rows):
        order_id = fake.uuid4()
        user_id = random.randint(1, 10000)
        amount = round(random.uniform(10, 1000), 2)
        status = random.choice(["PAID", "UNPAID", "CANCELLED"])
        records.append(
            {
                "dt": dt_str,
                "user_id": user_id,
                "order_id": order_id,
                "amount": amount,
                "status": status,
            }
        )
    return pd.DataFrame.from_records(records)


def inject_anomalies_for_today(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """在 t 日数据中注入重复 order_id 与 order_id 为空的记录。"""
    if df.empty:
        return df

    dt_str = target_date.isoformat()
    anomaly_records = []

    duplicated_order_ids = df["order_id"].head(3).tolist()
    for oid in duplicated_order_ids:
        anomaly_records.append(
            {
                "dt": dt_str,
                "user_id": random.randint(1, 10000),
                "order_id": oid,
                "amount": round(random.uniform(10, 1000), 2),
                "status": random.choice(["PAID", "UNPAID", "CANCELLED"]),
            }
        )

    for _ in range(2):
        anomaly_records.append(
            {
                "dt": dt_str,
                "user_id": random.randint(1, 10000),
                "order_id": None,
                "amount": round(random.uniform(10, 1000), 2),
                "status": random.choice(["PAID", "UNPAID", "CANCELLED"]),
            }
        )

    anomaly_df = pd.DataFrame.from_records(anomaly_records)
    return pd.concat([df, anomaly_df], ignore_index=True)


def main() -> None:
    engine = get_write_engine()
    create_table_if_not_exists(engine)

    if os.getenv("INIT_VALIDATION_TRUNCATE", "").lower() in ("1", "true", "yes"):
        truncate_validation_order_table_doris(engine)
        print("已按 INIT_VALIDATION_TRUNCATE 清空 dwd_trade_order_di。")

    today = date.today()
    yesterday = today - timedelta(days=1)

    base_count_yesterday = 1000
    anomaly_count_today = 5
    base_count_today = int(base_count_yesterday * 0.8) - anomaly_count_today

    if base_count_today <= 0:
        raise ValueError("基础数据量配置错误，请检查 base_count_yesterday 与 anomaly_count_today。")

    df_yesterday = generate_orders_for_date(yesterday, base_count_yesterday)
    df_today_base = generate_orders_for_date(today, base_count_today)
    df_today = inject_anomalies_for_today(df_today_base, today)

    # 列顺序与 Doris 表定义一致（dt, user_id, order_id, amount, status）
    col_order = ["dt", "user_id", "order_id", "amount", "status"]
    df_yesterday = df_yesterday[col_order]
    df_today = df_today[col_order]

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

    print("=== 数据生成完成（Apache Doris）===")
    print(f"日期 {yesterday} 插入记录数：{len(df_yesterday)}")
    print(f"日期 {today} 插入记录数：{len(df_today)}（含 {anomaly_count_today} 条刻意制造的异常）")
    print("目标表：dwd_trade_order_di")


if __name__ == "__main__":
    main()
