"""
一次性运行两类评测并输出合并摘要：

1) SQL：evaluation.run_benchmark — sql_query_result_accuracy
2) RAG 口径：evaluation.run_rag_benchmark — with_rag / without_rag 相似度与准确率

用法：
  python -m evaluation.run_all_metrics
  python -m evaluation.run_all_metrics --sql-json evaluation/golden_sql.json --rag-json evaluation/rag_golden.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    parser = argparse.ArgumentParser(description="合并运行 SQL + RAG 评测")
    parser.add_argument(
        "--sql-json",
        default=os.path.join(root, "evaluation", "golden_sql.json"),
    )
    parser.add_argument(
        "--rag-json",
        default=os.path.join(root, "evaluation", "rag_golden.json"),
    )
    parser.add_argument("--out", default="", help="合并 JSON 写入路径")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")

    sql_proc = subprocess.run(
        [sys.executable, "-m", "evaluation.run_benchmark", "--json", args.sql_json],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
    )
    rag_proc = subprocess.run(
        [sys.executable, "-m", "evaluation.run_rag_benchmark", "--json", args.rag_json],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
    )

    merged: dict = {
        "sql_benchmark": {"returncode": sql_proc.returncode, "stderr": sql_proc.stderr},
        "rag_benchmark": {"returncode": rag_proc.returncode, "stderr": rag_proc.stderr},
    }
    try:
        merged["sql"] = json.loads(sql_proc.stdout) if sql_proc.stdout.strip() else {}
    except json.JSONDecodeError:
        merged["sql_raw_stdout"] = sql_proc.stdout
    try:
        merged["rag"] = json.loads(rag_proc.stdout) if rag_proc.stdout.strip() else {}
    except json.JSONDecodeError:
        merged["rag_raw_stdout"] = rag_proc.stdout

    if sql_proc.returncode == 0 and rag_proc.returncode == 0:
        merged["headline"] = {
            "sql_query_result_accuracy_pct": (merged.get("sql") or {})
            .get("summary", {})
            .get("sql_query_result_accuracy"),
            "rag_accuracy_with_retrieval_pct": (merged.get("rag") or {})
            .get("with_rag", {})
            .get("accuracy_percent"),
            "rag_accuracy_without_retrieval_pct": (merged.get("rag") or {})
            .get("without_rag", {})
            .get("accuracy_percent"),
            "rag_lift_accuracy_points": (merged.get("rag") or {}).get(
                "rag_lift_accuracy_points"
            ),
        }

    out = json.dumps(merged, ensure_ascii=False, indent=2)
    print(out)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"已写入：{args.out}", file=sys.stderr)

    if sql_proc.returncode != 0 or rag_proc.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
