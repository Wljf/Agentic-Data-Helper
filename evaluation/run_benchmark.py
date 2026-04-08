"""
黄金集评测：用标准 SQL + 在 Doris 上执行得到的结果集，与模型生成的 SQL 及执行结果对比。

指标说明：
- result_match（主指标）：标准 SQL 与生成 SQL 在 Doris 上执行后的结果集是否一致（排序无关，数值容差）。
- sql_match（辅指标）：规范化后的 SQL 字符串是否一致（别名/格式不同也可能结果正确，故仅作参考）。

用法（项目根目录）：
  python -m evaluation.run_benchmark
  python -m evaluation.run_benchmark --json evaluation/golden_sql.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from agents.query_agent import run_query_agent_eval, run_query_agent_sql_only
from agents.tools import get_analytics_engine


def _normalize_sql(s: str) -> str:
    s = s.strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _norm_cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, Decimal):
        return round(float(v), 6)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return round(v, 6)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            return round(float(s), 6)
        except ValueError:
            return s
    return str(v).strip()


def _rows_to_sorted_tuples(raw: Sequence[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    normed = [tuple(_norm_cell(c) for c in row) for row in raw]
    return sorted(normed, key=lambda t: json.dumps(t, ensure_ascii=False, default=str))


def fetch_result_tuples(sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    """执行只读 SQL，返回 (列名, 数据行)。"""
    engine = get_analytics_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        rows = result.fetchall()
    data = [tuple(row) for row in rows]
    return [str(c) for c in cols], data


def result_sets_equivalent(
    a: Sequence[Tuple[Any, ...]], b: Sequence[Tuple[Any, ...]]
) -> bool:
    if len(a) != len(b):
        return False
    return _rows_to_sorted_tuples(a) == _rows_to_sorted_tuples(b)


def table_rows_to_tuples(table: Optional[List[List[Any]]]) -> Optional[List[Tuple[Any, ...]]]:
    """将 query_agent 返回的 [header, ...] 转为仅数据行的元组列表（值转为与 fetch 相近类型）。"""
    if not table or len(table) < 2:
        return None
    # 第一行为表头，数据行为字符串形式，评测时与数值比对用规范化
    data_rows = []
    for r in table[1:]:
        data_rows.append(tuple(r))
    return data_rows


def load_cases(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(case: Dict[str, Any]) -> Dict[str, Any]:
    q = case["question"]
    expected_sql = case["expected_sql"]
    force_pipeline = bool(case.get("force_pipeline", False))

    out: Dict[str, Any] = {
        "id": case.get("id"),
        "question": q,
        "expected_sql": expected_sql,
        "force_pipeline": force_pipeline,
    }

    exp_cols, exp_data = fetch_result_tuples(expected_sql)
    out["expected_columns"] = exp_cols
    out["expected_rowcount"] = len(exp_data)

    sql_only = case.get("sql_only_eval", True)
    if sql_only:
        answer_text, meta = run_query_agent_sql_only(q)
    else:
        answer_text, meta = run_query_agent_eval(q, force_pipeline=force_pipeline)
    out["answer_preview"] = (answer_text or "")[:500]
    out["eval_meta"] = {
        "path": meta.get("path"),
        "sql_only_eval": sql_only,
        "generated_sql": meta.get("generated_sql"),
        "safe_passed": meta.get("safe_passed"),
        "execution_ok": meta.get("execution_ok"),
    }

    gen_sql = meta.get("generated_sql") or meta.get("executed_sql")
    out["generated_sql"] = gen_sql

    sql_match = False
    if gen_sql and isinstance(gen_sql, str):
        sql_match = _normalize_sql(gen_sql) == _normalize_sql(expected_sql)
    out["sql_match"] = sql_match

    result_match = False
    gen_data: Optional[List[Tuple[Any, ...]]] = None
    if gen_sql and meta.get("safe_passed") and meta.get("execution_ok"):
        try:
            if _normalize_sql(gen_sql).startswith("select") or _normalize_sql(
                gen_sql
            ).startswith("with"):
                _, gen_data = fetch_result_tuples(gen_sql)
        except Exception as e:
            out["generated_fetch_error"] = str(e)

    if gen_data is None and meta.get("rows"):
        # 来自 pipeline 的字符串表格，尽力对比
        t = table_rows_to_tuples(meta.get("rows"))
        if t is not None:
            gen_data = t

    if gen_data is not None:
        result_match = result_sets_equivalent(exp_data, gen_data)

    out["result_match"] = result_match
    out["generated_rowcount"] = len(gen_data) if gen_data is not None else None

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="黄金集 SQL 结果评测")
    parser.add_argument(
        "--json",
        default=os.path.join(
            os.path.dirname(__file__), "golden_sql.json"
        ),
        help="黄金集 JSON 路径",
    )
    parser.add_argument(
        "--out",
        default="",
        help="将详细结果写入该 JSON 文件（可选）",
    )
    args = parser.parse_args()

    bundle = load_cases(args.json)
    cases = bundle.get("cases", [])
    results: List[Dict[str, Any]] = []

    for c in cases:
        try:
            results.append(run_one(c))
        except Exception as e:
            results.append(
                {
                    "id": c.get("id"),
                    "error": str(e),
                    "question": c.get("question"),
                }
            )

    n = len(results)
    rm = sum(1 for r in results if r.get("result_match"))
    sm = sum(1 for r in results if r.get("sql_match"))
    ra = round(100.0 * rm / n, 2) if n else 0.0
    sa = round(100.0 * sm / n, 2) if n else 0.0
    summary = {
        "total": n,
        "sql_query_result_accuracy": ra,
        "sql_string_match_accuracy": sa,
        "result_match_count": rm,
        "sql_match_count": sm,
        "note": "sql_query_result_accuracy：生成 SQL 与标准 SQL 在 Doris 上执行结果集是否一致（主指标）。"
        " sql_string_match_accuracy：规范化后 SQL 文本是否完全一致（辅指标）。",
        "aliases": {
            "result_accuracy": ra,
            "sql_string_accuracy": sa,
        },
    }

    print(json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "details": results, "source": args.json},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n已写入：{args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
