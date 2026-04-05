"""
BIRD Text-to-SQL Evaluation Harness

Scores predicted SQL against gold SQL by executing both on the 
database and comparing result sets. Execution accuracy = fraction 
of tasks where predicted result set matches gold result set.

Usage:
    python -m eval.harness --data_dir data/dev --predictions results/predictions.json
    python -m eval.harness --data_dir data/dev --predictions results/predictions.json --output results/detailed.json
"""

import json
import sqlite3
import argparse
import os
import sys
from pathlib import Path
from typing import Optional


# ── Data Loading ──────────────────────────────────────────────────

def load_dev_tasks(data_dir: str) -> list[dict]:
    """Load BIRD dev.json and return list of task dicts."""
    dev_json_path = os.path.join(data_dir, "dev.json")
    with open(dev_json_path, "r") as f:
        tasks = json.load(f)
    
    # Normalize keys - BIRD uses 'evidence' for hints
    for i, task in enumerate(tasks):
        task["task_idx"] = i
        if "evidence" in task:
            task["hint"] = task["evidence"]
    
    return tasks


def get_db_path(data_dir: str, db_id: str) -> str:
    """Get path to SQLite database file for a given db_id."""
    # BIRD structure: dev_databases/{db_id}/{db_id}.sqlite
    db_path = os.path.join(data_dir, "dev_databases", db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    return db_path


def load_predictions(predictions_path: str) -> dict:
    """
    Load predictions file. Supports two formats:
    
    1. Dict mapping task index (as string) to SQL string:
       {"0": "SELECT ...", "1": "SELECT ...", ...}
    
    2. List of SQL strings (indexed by position):
       ["SELECT ...", "SELECT ...", ...]
    """
    with open(predictions_path, "r") as f:
        preds = json.load(f)
    
    if isinstance(preds, list):
        return {str(i): sql for i, sql in enumerate(preds)}
    return preds


# ── SQL Execution ─────────────────────────────────────────────────

def execute_sql(db_path: str, sql: str, timeout: float = 10.0) -> tuple[Optional[list[tuple]], Optional[str]]:
    """
    Execute SQL against a SQLite database.
    
    Returns:
        (results, error) - results is a list of tuples if successful,
                          error is a string if execution failed.
    """
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
    """
    Normalize a result set for comparison.
    - Convert all values to strings (handles type mismatches)
    - Sort rows (order-insensitive comparison)
    """
    def normalize_value(v):
        if v is None:
            return "NULL"
        # Handle numeric equivalence: 1 == 1.0
        if isinstance(v, (int, float)):
            # If it's a float that's actually an integer (e.g. 3.0), treat as int
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
    pred_norm = normalize_result_set(pred_results)
    gold_norm = normalize_result_set(gold_results)
    return pred_norm == gold_norm


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_task(
    task: dict,
    predicted_sql: str,
    data_dir: str,
    timeout: float = 10.0
) -> dict:
    """
    Evaluate a single task. Returns a result dict with:
    - match: bool
    - error_type: None | 'pred_error' | 'gold_error' | 'mismatch'
    - pred_error: error message if predicted SQL failed
    - gold_error: error message if gold SQL failed
    """
    db_id = task["db_id"]
    gold_sql = task["SQL"]
    db_path = get_db_path(data_dir, db_id)
    
    # Execute gold SQL
    gold_results, gold_error = execute_sql(db_path, gold_sql, timeout)
    if gold_error:
        # Gold SQL failed - this is a dataset issue, skip
        return {
            "task_idx": task["task_idx"],
            "db_id": db_id,
            "question": task["question"],
            "hint": task.get("hint", ""),
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "match": False,
            "error_type": "gold_error",
            "gold_error": gold_error,
            "pred_error": None,
            "skipped": True,
        }
    
    # Execute predicted SQL
    pred_results, pred_error = execute_sql(db_path, predicted_sql, timeout)
    if pred_error:
        return {
            "task_idx": task["task_idx"],
            "db_id": db_id,
            "question": task["question"],
            "hint": task.get("hint", ""),
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "match": False,
            "error_type": "pred_error",
            "pred_error": pred_error,
            "gold_error": None,
            "skipped": False,
        }
    
    # Compare result sets
    match = compare_results(pred_results, gold_results)
    
    return {
        "task_idx": task["task_idx"],
        "db_id": db_id,
        "question": task["question"],
        "hint": task.get("hint", ""),
        "gold_sql": gold_sql,
        "predicted_sql": predicted_sql,
        "match": match,
        "error_type": None if match else "mismatch",
        "pred_error": None,
        "gold_error": None,
        "skipped": False,
    }


def run_evaluation(
    data_dir: str,
    predictions_path: str,
    output_path: Optional[str] = None,
    timeout: float = 10.0,
    verbose: bool = True,
) -> dict:
    """
    Run full evaluation on BIRD dev set.
    
    Returns summary dict with execution accuracy and per-task results.
    """
    tasks = load_dev_tasks(data_dir)
    predictions = load_predictions(predictions_path)
    
    results = []
    correct = 0
    total = 0
    skipped = 0
    error_counts = {"pred_error": 0, "mismatch": 0, "gold_error": 0}
    db_scores = {}  # per-database accuracy
    
    for task in tasks:
        idx_str = str(task["task_idx"])
        
        if idx_str not in predictions:
            if verbose:
                print(f"  WARNING: No prediction for task {idx_str}, skipping")
            continue
        
        predicted_sql = predictions[idx_str]
        result = evaluate_task(task, predicted_sql, data_dir, timeout)
        results.append(result)
        
        if result["skipped"]:
            skipped += 1
            continue
        
        total += 1
        if result["match"]:
            correct += 1
        else:
            error_counts[result["error_type"]] += 1
        
        # Track per-database accuracy
        db_id = result["db_id"]
        if db_id not in db_scores:
            db_scores[db_id] = {"correct": 0, "total": 0}
        db_scores[db_id]["total"] += 1
        if result["match"]:
            db_scores[db_id]["correct"] += 1
    
    # Compute per-database accuracy
    db_accuracy = {
        db: round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0
        for db, s in sorted(db_scores.items())
    }
    
    accuracy = round(correct / total, 4) if total > 0 else 0
    
    summary = {
        "execution_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "error_breakdown": error_counts,
        "per_database_accuracy": db_accuracy,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"BIRD Execution Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Skipped (gold errors): {skipped}")
        print(f"  Pred SQL errors:       {error_counts['pred_error']}")
        print(f"  Result mismatches:     {error_counts['mismatch']}")
        print(f"\nPer-database accuracy:")
        for db, acc in db_accuracy.items():
            db_total = db_scores[db]["total"]
            db_correct = db_scores[db]["correct"]
            print(f"  {db:30s} {acc:.4f} ({db_correct}/{db_total})")
        print(f"{'='*50}")
    
    output = {"summary": summary, "results": results}
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"\nDetailed results saved to: {output_path}")
    
    return output


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BIRD Text-to-SQL Evaluation Harness")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to BIRD dev data directory")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save detailed results JSON")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="SQL execution timeout in seconds")
    
    args = parser.parse_args()
    run_evaluation(args.data_dir, args.predictions, args.output, args.timeout)


if __name__ == "__main__":
    main()