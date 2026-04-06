"""
Reward function for GRPO training on BIRD text-to-SQL.

Executes generated SQL against the database and compares result sets
to gold SQL. Returns scalar reward.

This module is self-contained (no imports from eval/ or scaffold/)
so it can run inside the Modal training container.

Usage:
    from rl.reward import compute_reward, compute_rewards_batch

    reward = compute_reward(generated_sql, gold_sql, db_path)
"""

import sqlite3
from typing import Optional


def execute_sql(db_path: str, sql: str, timeout: float = 5.0) -> tuple[Optional[list[tuple]], Optional[str]]:
    """Execute SQL against a SQLite database. Returns (results, error)."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("PRAGMA journal_mode=wal;")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def normalize_result_set(results: list[tuple]) -> list[tuple]:
    """Normalize result set for comparison (order-insensitive, type-normalized)."""
    def normalize_value(v):
        if v is None:
            return "NULL"
        if isinstance(v, (int, float)):
            try:
                if float(v) == int(float(v)):
                    return str(int(float(v)))
            except (ValueError, OverflowError):
                pass
            return str(v)
        return str(v).strip().lower()

    normalized = [
        tuple(normalize_value(v) for v in row)
        for row in results
    ]
    return sorted(normalized)


def compare_results(pred_results: list[tuple], gold_results: list[tuple]) -> bool:
    """Compare two result sets, order-insensitive."""
    return normalize_result_set(pred_results) == normalize_result_set(gold_results)


def compute_reward(
    generated_sql: str,
    gold_sql: str,
    db_path: str,
    timeout: float = 5.0,
) -> float:
    """
    Compute reward for a generated SQL query.

    Returns:
        1.0 — result sets match (correct)
        0.1 — generated SQL executes but returns wrong results (partial credit)
        0.0 — generated SQL errors or times out
    """
    # Execute gold SQL
    gold_results, gold_error = execute_sql(db_path, gold_sql, timeout)
    if gold_error:
        # Gold SQL failed — dataset issue, give no signal
        return 0.0

    # Execute generated SQL
    pred_results, pred_error = execute_sql(db_path, generated_sql, timeout)
    if pred_error:
        return 0.0

    # Compare result sets
    if compare_results(pred_results, gold_results):
        return 1.0

    # Valid SQL but wrong answer — partial credit
    return 0.1


def compute_rewards_batch(
    generated_sqls: list[str],
    gold_sqls: list[str],
    db_paths: list[str],
    timeout: float = 5.0,
) -> list[float]:
    """Compute rewards for a batch of generated SQL queries."""
    return [
        compute_reward(gen, gold, db_path, timeout)
        for gen, gold, db_path in zip(generated_sqls, gold_sqls, db_paths)
    ]
