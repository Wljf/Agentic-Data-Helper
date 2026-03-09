# 电商数仓业务口径定义

> 本文件由数仓与 BI 约定，用于 Agentic RAG 检索。向量化后存入 ChromaDB。

## 新老客定义

- **口径名称**：新用户 / 老用户（新老客）
- **业务定义**：
  - **新用户（新客）**：在统计周期内，**首次**在该业务域产生订单（或首次支付）的用户。
  - **老用户（老客）**：在统计周期内产生订单，且**非**首次下单的用户（即历史已有订单）。
- **技术口径**：
  - 以「用户首次下单日期」判断：若用户在某日的「首次下单日期」等于该日，则该日该用户为新客；否则为老客。
  - 数仓中常用字段：`is_new_user`（1=新客，0=老客），存在于 **DWS 层用户汇总表**（如 `dws_user_trade_summary_nd`）中。

## 客单价（ATV）

- **口径名称**：客单价（Average Transaction Value）
- **业务定义**：在统计周期内，平均每笔订单的金额（或平均每用户的交易金额，视口径而定）。
- **常见两种计算方式**：
  1. **按订单**：ATV = 总交易金额（GMV）/ 订单笔数。
  2. **按用户（人均）**：某类用户（如新客）的 ATV = 该类用户的总交易金额 / 该类用户数（去重）。
- **技术口径**：
  - 总交易金额来自订单明细或汇总表的 `pay_amount` / `total_amount` 等字段。
  - 订单数、用户数来自 COUNT 聚合；新老客 ATV 需先按 `is_new_user` 分组再分别计算。

## GMV 口径

- **口径名称**：GMV（Gross Merchandise Volume，成交总额）
- **业务定义**：在统计周期内，所有**已支付**订单的金额总和（不含取消、未支付）。
- **技术口径**：
  - 数据来源：订单明细表（如 `dwd_trade_order_detail_di`）的 `pay_amount` 字段，按订单去重后按订单维度汇总金额，或直接对明细行 `SUM(pay_amount)`（需注意一单多行时按 order_id 去重再汇总，或按业务约定是否按行汇总）。
  - 本项目中 DWD 表为订单明细行（一单多行），GMV = SUM(pay_amount)，或按 order_id 去重后按订单汇总金额；ADS 看板表 `ads_sales_dashboard_di` 中已有 `gmv` 字段，可直接使用。

## 表层级与用途

- **ODS**：原始日志、原始订单等，按天分区；表名示例 `ods_log_user_action_di`。
- **DWD**：明细层，订单明细表如 `dwd_trade_order_detail_di`，字段含 order_id, user_id, sku_id, pay_amount, dt。
- **DWS**：轻度汇总，用户粒度汇总如 `dws_user_trade_summary_nd`，含 user_id, is_new_user, total_amount, order_count, dt。
- **ADS**：应用层看板，如 `ads_sales_dashboard_di`，含 dt, gmv, order_count, new_user_count, old_user_count, atv_new, atv_old。
