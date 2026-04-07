# Data Agent 数据助手

一个面向数据分析与数据开发场景的本地 Demo，支持两类核心能力：

- `数据验证`：检查主键唯一性、非空性，以及分区数据量波动
- `查数找数`：基于 `LangGraph + Agentic RAG + Text-to-SQL` 完成复杂业务问题拆解、口径检索、SQL 生成与结果回显

项目重点解决以下问题：

- 业务口径散落在文档里，纯 Text-to-SQL 容易“猜错口径”
- 复杂查数问题往往包含多个隐含子任务，单轮问答命中率低
- 大模型可能生成危险 SQL，存在幻觉和安全风险

---

## 项目亮点

- 设计并实现 `ODS / DWD / DWS / ADS` 四层电商数仓模拟链路，支持连续多天数据生成，并通过大促倍数模拟波动
- 使用 `ChromaDB + HuggingFace Embeddings` 构建离线业务知识库，沉淀新老客、ATV、GMV 等核心口径
- 使用 `LangGraph` 将复杂查数流程重构为标准状态机：问题拆解、子问题重写、迭代检索、反思决策、全局聚合
- 对 SQL 生成与执行增加 Prompt 层和代码层双重防护，仅允许单条只读查询执行
- 支持 SQL 执行结果回显，用户最终可以看到“口径说明 + SQL + 查询结果”

---

## 技术栈

- 语言：`Python`
- Web：`Flask`
- 数据库：`Apache Doris`（MySQL 协议）、`SQLAlchemy`
- 数据处理：`Pandas`、`Faker`
- LLM / Agent：`LangChain`、`LangGraph`、`OpenAI Compatible API / DeepSeek`
- 向量检索：`ChromaDB`
- Embedding：`sentence-transformers`、`HuggingFaceEmbeddings`
- 前端：`Bootstrap 5`、`marked.js`

---

## 系统架构

### 1. 在线服务入口

- 前端通过 `templates/index.html` 提供聊天式交互界面
- 后端通过 `app.py` 暴露 `/chat` 接口
- 根据 `scene` 路由到不同场景：
  - `validation` -> `agents/validation_agent.py`
  - `query` -> `agents/query_agent.py`

### 2. 数据验证链路

`validation_agent.py` 的处理流程如下：

1. 用 LLM 从自然语言里抽取表名、主键字段、日期参数
2. 调用 `check_pk_tool` 检查主键唯一性 / 非空性
3. 调用 `check_volume_tool` 对比 t 日与 t-1 日数据量
4. 汇总为结构化 Markdown 报告

### 3. 查数找数链路

`query_agent.py` 会根据问题复杂度做分流：

- 简单问题：走原有 `RAG + 元数据 + Text-to-SQL`
- 复杂问题：走 `agents/agentic_query.py` 中的 Agentic RAG 图

复杂问题的 LangGraph 状态机如下：

```text
START
  -> node_decompose
  -> node_refine
  -> node_retrieve
  -> node_reflect
     -> node_retrieve      (信息不足且轮次 < 3)
     -> node_finalize_sub  (可回答，或达到最大轮次)
  -> node_refine           (还有下一个子问题)
  -> node_aggregate
END
```

核心思路：

- `node_decompose`：把复杂问题拆成多个独立子问题
- `node_refine`：结合前序结论做指代消解
- `node_retrieve`：查询 ChromaDB 业务口径、数据库表列表和表结构元数据
- `node_reflect`：判断当前信息是否足够回答，若不足则生成下一轮检索问题
- `node_finalize_sub`：收敛单个子问题结论，必要时附加免责声明
- `node_aggregate`：汇总所有子结论，生成最终 SQL 和口径解释

### 4. SQL 安全机制

项目对 SQL 做了双重防护：

- Prompt 层：明确要求只生成单条只读 SQL，忽略注入式指令
- 代码层：执行前再次校验
  - 仅允许 `SELECT` 或 `WITH ... SELECT`
  - 允许末尾单个分号
  - 禁止中间分号（多语句）
  - 禁止危险关键字，如 `INSERT / UPDATE / DELETE / DROP / ALTER`

---

## 目录结构

```text
data_agent_project/
├─ app.py
├─ config.py
├─ requirements.txt
├─ agents/
│  ├─ agentic_query.py
│  ├─ query_agent.py
│  ├─ tools.py
│  └─ validation_agent.py
├─ data_mock/
│  ├─ doris_ddls.py
│  ├─ doris_engine.py
│  ├─ generate_data.py
│  └─ init_warehouse.py
├─ docs/
│  └─ doris-setup.md
├─ knowledge_base/
│  ├─ build_rag.py
│  ├─ definitions.md
│  └─ business_definitions.md
└─ templates/
   └─ index.html
```

---

## 快速开始

### 1. 安装依赖

建议使用虚拟环境：

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env`，至少包含以下配置：

```env
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
# Apache Doris（MySQL 协议，连接 FE 默认端口 9030）
# 初始化脚本需要建表/写入权限；生产环境 Agent 建议使用只读账号，见 docs/doris-setup.md
DORIS_DB_URL=mysql+pymysql://user:password@127.0.0.1:9030/data_agent
CHROMA_PERSIST_DIR=./chroma_db
FLASK_DEBUG=True
```

说明：

- `OPENAI_API_BASE` 支持 OpenAI 兼容接口，这里默认可接 DeepSeek
- `DORIS_DB_URL` 为 Web 应用、Agent 查询与 **数仓初始化脚本** 的默认目标库；部署与权限详见 [docs/doris-setup.md](docs/doris-setup.md)

### 3. 初始化数仓数据

**请先完成 Doris 部署与库表权限配置**（见 [docs/doris-setup.md](docs/doris-setup.md)），然后在项目根目录执行：

```bash
python -m data_mock.init_warehouse
```

可选：重复执行前清空四张业务表再导入：

```bash
INIT_WAREHOUSE_TRUNCATE=true python -m data_mock.init_warehouse
```

该脚本会在 **Doris** 中创建并写入以下表：

- `ods_log_user_action_di`
- `dwd_trade_order_detail_di`
- `dws_user_trade_summary_nd`
- `ads_sales_dashboard_di`

数据验证场景（主键/数据量波动 Demo）需额外导入：

```bash
python -m data_mock.generate_data
```

可选：`INIT_VALIDATION_TRUNCATE=true python -m data_mock.generate_data` 仅清空 `dwd_trade_order_di` 后重导。

### 4. 构建知识库

```bash
python -m knowledge_base.build_rag
```

该脚本会将以下业务文档向量化并写入本地 `ChromaDB`：

- `knowledge_base/definitions.md`
- `knowledge_base/business_definitions.md`

### 5. 启动服务

```bash
python app.py
```

浏览器访问：

`http://127.0.0.1:5000`

---

## 如何重新生成数据

如果你想在 **Doris** 中重建数仓与验证样例数据，建议按以下顺序：

### 1. 清空并重新导入数仓（可选）

```bash
INIT_WAREHOUSE_TRUNCATE=true python -m data_mock.init_warehouse
INIT_VALIDATION_TRUNCATE=true python -m data_mock.generate_data
```

### 2. 重建向量知识库（可选）

```bash
python -m knowledge_base.build_rag
```

---

## 示例问句

### 数据验证

- `帮我检查今天和昨天 dwd_trade_order_di 表的主键和数据量是否正常`
- `检查昨天订单表的主键是否有重复，顺便看一下今天和昨天的数据量变化`

### 查数找数

- `帮我对比一下昨天新老用户的客单价差异，并给出 SQL`
- `GMV 的业务口径是什么？`
- `帮我看一下昨天大促期间的大盘销售表现`
- `哪张表有新老客标签和交易金额字段？`

---

## 离线数据与业务口径说明

### 数仓分层

- `ODS`：原始用户行为日志
- `DWD`：订单明细
- `DWS`：用户粒度交易汇总，包含 `is_new_user`
- `ADS`：大盘销售看板

### 核心业务口径

项目已内置以下业务定义：

- 新客 / 老客定义
- 客单价（ATV）按订单 / 按用户两种常见口径
- GMV 统计口径

这些内容位于 `knowledge_base/business_definitions.md`，在线查询时会通过向量检索参与回答。

---

## 面向简历的项目表述

可用于简历中的一句话总结：

> 设计并落地一套基于 `LangGraph + Agentic RAG` 的智能数据助手，围绕复杂查数场景实现问题拆解、多轮反思检索、业务口径对齐、只读 SQL 生成与结果回显，降低 Text-to-SQL 幻觉并提升复杂分析问答准确率。

---

## 注意事项

- 不要将 `.env`、本地数据库文件、`chroma_db/` 等内容提交到 Git
- 当前 Demo 使用的是 `SQLite`，生产环境可替换为只读 `MySQL`
- 如遇接口报错，前端会展示后端 Traceback，便于快速排查

---

## 后续可扩展方向

- 接入真实 MySQL / Hive 元数据服务
- 对知识库进行分段切片，而不是整篇 Markdown 入库
- 增加 SQL 执行计划、耗时和命中表字段等可观测信息
- 引入会话级记忆与问题上下文复用

