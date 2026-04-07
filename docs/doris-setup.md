# Apache Doris 配置说明

本文说明如何部署/接入 Doris、创建库表权限，以及与本项目 `DORIS_DB_URL`、数据初始化脚本的配合方式。

---

## 1. 与本项目的关系

| 组件 | 作用 |
|------|------|
| **FE（Frontend）** | 对外提供 **MySQL 协议** 查询入口，默认端口 **9030**。 |
| **BE（Backend）** | 存储与计算节点，由集群内部协调。 |
| **本项目** | 通过 `mysql+pymysql://...` 连接 **FE 的 9030**，执行建表、`INSERT`、`SELECT`、元数据查询（`information_schema`）等。 |

`config.DORIS_DB_URL` 应指向：**已创建好的业务库**，例如 `.../data_agent`。

---

## 2. 部署 Doris（概要）

请优先以 [Apache Doris 官方文档](https://doris.apache.org/docs/gettingStarted/quick-start) 为准，选择：

- **集群 / 生产**：按官方指引部署多 FE、多 BE，并配置副本与监控。
- **本地体验**：使用官方提供的快速体验方式（如 Docker、一键脚本等），确保至少 **1 个 FE + 1 个 BE** 处于可用状态。

**开发环境注意：**

- 本仓库 `data_mock/doris_ddls.py` 中表属性为 `"replication_num" = "1"`，适用于 **单副本 / 本地单节点** 场景。
- 若集群要求副本数 ≥ 3，需把 DDL 中的 `replication_num` 改为与集群策略一致（并保证有足够 BE）。

---

## 3. 创建数据库与账号

使用任意 MySQL 客户端连接 **FE**（示例：`mysql -h <FE_HOST> -P 9030 -u root`），执行：

```sql
CREATE DATABASE IF NOT EXISTS data_agent;
```

### 3.1 初始化脚本用账号（需建表、写入、TRUNCATE）

初始化脚本：`python -m data_mock.init_warehouse`、`python -m data_mock.generate_data`。

建议授予业务库下 **建表、读写、清空** 等权限（具体以你们安全规范为准），例如：

```sql
CREATE USER 'dwh_writer'@'%' IDENTIFIED BY 'your_strong_password';
GRANT SELECT, INSERT, UPDATE, DELETE, ALTER, CREATE, DROP ON data_agent.* TO 'dwh_writer'@'%';
```

> 若你希望最小权限，可按实际报错逐项补权；`CREATE TABLE`、`INSERT`、`TRUNCATE` 在不同版本上可能映射到 `CREATE`、`INSERT`、`DELETE` 等。

### 3.2 应用 / Agent 只读账号（推荐生产使用）

Web 与 Agent 仅需查询时，使用只读账号配置 `DORIS_DB_URL`：

```sql
CREATE USER 'dwh_readonly'@'%' IDENTIFIED BY 'your_strong_password';
GRANT SELECT ON data_agent.* TO 'dwh_readonly'@'%';
```

本地开发也可 **暂时** 使用与初始化相同的写账号，减少配置项。

---

## 4. 配置项目 `.env`

在项目根目录 `.env` 中设置（将主机、端口、库名、账号改为你的环境）：

```env
# 示例：初始化脚本 + 本地联调（写权限）
DORIS_DB_URL=mysql+pymysql://dwh_writer:your_strong_password@127.0.0.1:9030/data_agent

# 若应用与初始化分开账号，应用可单独使用只读 URL（需改代码或第二套配置，当前代码仅读取 DORIS_DB_URL）
# DORIS_DB_URL=mysql+pymysql://dwh_readonly:...@127.0.0.1:9030/data_agent
```

**说明：**

- 协议为 **`mysql+pymysql`**，与 SQLAlchemy、pandas `to_sql` 兼容。
- 端口一般为 **9030**（FE 查询端口），不是 HTTP 8030 等管理端口。
- 路径中的 **库名** 必须与上文 `CREATE DATABASE` 一致。

---

## 5. 导入数仓与验证数据

在项目根目录执行（需已安装依赖且 `DORIS_DB_URL` 可用）：

```bash
# 四层数仓（ODS/DWD/DWS/ADS）
python -m data_mock.init_warehouse

# 数据验证场景表 dwd_trade_order_di（含主键异常样例）
python -m data_mock.generate_data
```

**重复执行前清空表（可选）：**

```bash
# 清空 ODS/DWD/DWS/ADS 四表后再导入
INIT_WAREHOUSE_TRUNCATE=true python -m data_mock.init_warehouse

# 仅清空 dwd_trade_order_di
INIT_VALIDATION_TRUNCATE=true python -m data_mock.generate_data
```

---

## 6. 常见问题

| 现象 | 排查方向 |
|------|----------|
| 无法连接 9030 | FE 是否启动、防火墙、主机/端口是否填错。 |
| `replication_num` / 副本相关报错 | BE 数量与副本数是否匹配；开发环境可保持 DDL 中 `replication_num=1` 与单 BE。 |
| 建表失败 | 账号是否有 `CREATE`；库名是否存在。 |
| `TRUNCATE` 失败 | 账号是否有对应权限；表是否存在。 |
| 应用查询为空 | 是否已执行初始化脚本；`DORIS_DB_URL` 是否指向同一库。 |

---

## 7. 参考链接

- [Apache Doris 官方文档](https://doris.apache.org/docs/)
- [SQLAlchemy MySQL / PyMySQL](https://docs.sqlalchemy.org/en/20/dialects/mysql.html)
