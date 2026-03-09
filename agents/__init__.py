"""
agents 包：

用于存放不同业务场景下的 Agent 封装逻辑，例如：
- validation_agent：数据验证场景
- query_agent：查数找数 / RAG / Text-to-SQL 场景

在 app.py 中会通过 run_validation_agent / run_query_agent 两个函数作为统一入口。
"""

