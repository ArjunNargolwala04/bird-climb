"""
Quick smoke tests for the eval harness using real BIRD dev data.
Run: python -m eval.test_harness
"""

import json
import os
import tempfile
import sys

from eval.harness import (
    load_dev_tasks,
    get_db_path,
    load_predictions,
    execute_sql,
    normalize_result_set,
    compare_results,
    evaluate_task,
    run_evaluation,
)
from eval.analyze import categorize_error, analyze_results, compare_runs
from eval.tracker import log_experiment, load_experiments

DATA_DIR = "data/dev"
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


# ── 1. load_dev_tasks ────────────────────────────────────────────
print("\n=== load_dev_tasks ===")
tasks = load_dev_tasks(DATA_DIR)
check("loads list", isinstance(tasks, list))
check("1534 tasks", len(tasks) == 1534, f"got {len(tasks)}")
check("task_idx added", tasks[0].get("task_idx") == 0)
check("hint normalized", "hint" in tasks[0])
check("has required keys", all(k in tasks[0] for k in ["db_id", "question", "SQL"]))

# ── 2. get_db_path ───────────────────────────────────────────────
print("\n=== get_db_path ===")
db_path = get_db_path(DATA_DIR, "california_schools")
check("returns path", db_path.endswith(".sqlite"))
check("file exists", os.path.exists(db_path))

try:
    get_db_path(DATA_DIR, "nonexistent_db")
    check("raises on missing db", False, "no exception raised")
except FileNotFoundError:
    check("raises on missing db", True)

# ── 3. execute_sql ───────────────────────────────────────────────
print("\n=== execute_sql ===")
db_path = get_db_path(DATA_DIR, "california_schools")

results, err = execute_sql(db_path, "SELECT COUNT(*) FROM frpm")
check("valid SQL returns results", results is not None and err is None)
check("returns list of tuples", isinstance(results, list) and isinstance(results[0], tuple))

results, err = execute_sql(db_path, "SELECT * FROM nonexistent_table")
check("bad table returns error", results is None and err is not None)
check("error mentions table", "no such table" in err.lower())

results, err = execute_sql(db_path, "SELEKT broken")
check("syntax error returns error", results is None and err is not None)

# ── 4. normalize_result_set ──────────────────────────────────────
print("\n=== normalize_result_set ===")

# Int/float equivalence
check("int/float equiv", normalize_result_set([(1,)]) == normalize_result_set([(1.0,)]))

# NULL handling
check("NULL handling", normalize_result_set([(None,)]) == [("NULL",)])

# String normalization (lowercase, strip)
check("string lower+strip", normalize_result_set([("  Hello ",)]) == [("hello",)])

# Order insensitivity
check("order insensitive",
      normalize_result_set([(2,), (1,)]) == normalize_result_set([(1,), (2,)]))

# Multi-column
check("multi-column",
      normalize_result_set([("a", 1), ("b", 2)]) == normalize_result_set([("b", 2), ("a", 1)]))

# ── 5. compare_results ──────────────────────────────────────────
print("\n=== compare_results ===")
check("identical match", compare_results([(1, "a")], [(1, "a")]))
check("order doesn't matter", compare_results([(1,), (2,)], [(2,), (1,)]))
check("int/float match", compare_results([(1,)], [(1.0,)]))
check("mismatch detected", not compare_results([(1,)], [(2,)]))
check("empty sets match", compare_results([], []))
check("empty vs non-empty", not compare_results([], [(1,)]))

# ── 6. evaluate_task (gold == pred should match) ─────────────────
print("\n=== evaluate_task (gold SQL as prediction) ===")
# Use gold SQL as prediction — should always match
task0 = tasks[0]
result = evaluate_task(task0, task0["SQL"], DATA_DIR)
check("gold==pred matches", result["match"] is True, f"error_type={result.get('error_type')}")
check("no errors", result["pred_error"] is None and result["gold_error"] is None)

# Test a few more tasks across different databases
test_indices = [0, 100, 500, 1000, 1500]
all_gold_match = True
for idx in test_indices:
    if idx >= len(tasks):
        continue
    t = tasks[idx]
    r = evaluate_task(t, t["SQL"], DATA_DIR)
    if not r["match"] and not r.get("skipped"):
        all_gold_match = False
        print(f"    gold mismatch at task {idx}: {r.get('error_type')} {r.get('pred_error', '')}")
check("gold SQL matches itself on 5 tasks", all_gold_match)

# ── 7. evaluate_task (wrong SQL) ─────────────────────────────────
print("\n=== evaluate_task (wrong predictions) ===")
result = evaluate_task(task0, "SELECT 999999", DATA_DIR)
check("wrong answer detected", result["match"] is False)
check("error_type=mismatch", result["error_type"] == "mismatch")

result = evaluate_task(task0, "SELEKT broken", DATA_DIR)
check("syntax error detected", result["match"] is False)
check("error_type=pred_error", result["error_type"] == "pred_error")

# ── 8. load_predictions (both formats) ───────────────────────────
print("\n=== load_predictions ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump({"0": "SELECT 1", "1": "SELECT 2"}, f)
    dict_path = f.name

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(["SELECT 1", "SELECT 2"], f)
    list_path = f.name

preds_dict = load_predictions(dict_path)
preds_list = load_predictions(list_path)
check("dict format", preds_dict == {"0": "SELECT 1", "1": "SELECT 2"})
check("list format", preds_list == {"0": "SELECT 1", "1": "SELECT 2"})
os.unlink(dict_path)
os.unlink(list_path)

# ── 9. run_evaluation (small test) ──────────────────────────────
print("\n=== run_evaluation (10 tasks, gold SQL) ===")
# Create predictions from gold SQL for first 10 tasks
gold_preds = {str(t["task_idx"]): t["SQL"] for t in tasks[:10]}
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(gold_preds, f)
    gold_pred_path = f.name

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    output_path = f.name

output = run_evaluation(DATA_DIR, gold_pred_path, output_path, verbose=False)
check("returns dict", isinstance(output, dict) and "summary" in output)
acc = output["summary"]["execution_accuracy"]
check(f"gold accuracy ~1.0", acc >= 0.9, f"got {acc}")
check("output file written", os.path.exists(output_path))

# Verify output file is valid JSON
with open(output_path) as f:
    saved = json.load(f)
check("saved output has results", "results" in saved and len(saved["results"]) == 10)

os.unlink(gold_pred_path)

# ── 10. analyze.py ───────────────────────────────────────────────
print("\n=== analyze_results ===")
analysis = analyze_results(output_path)
check("has error_categories", "error_categories" in analysis)
check("has db_errors", "db_errors" in analysis)

# ── 11. categorize_error ─────────────────────────────────────────
print("\n=== categorize_error ===")
check("correct", categorize_error({"match": True}) == "correct")
check("syntax_error", categorize_error({"match": False, "pred_error": "near syntax error"}) == "syntax_error")
check("no_such_table", categorize_error({"match": False, "pred_error": "no such table: foo"}) == "no_such_table")
check("no_such_column", categorize_error({"match": False, "pred_error": "no such column: bar"}) == "no_such_column")
check("wrong_answer", categorize_error({"match": False, "pred_error": ""}) == "wrong_answer")
check("gold_error", categorize_error({"match": False, "skipped": True}) == "gold_error")

os.unlink(output_path)

# ── 12. tracker.py ───────────────────────────────────────────────
print("\n=== tracker ===")
with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
    tracker_path = f.name

log_experiment("test_run", {"model": "test"}, 0.5, 5, 10, "smoke test", experiments_file=tracker_path)
records = load_experiments(tracker_path)
check("logged 1 record", len(records) == 1)
check("name correct", records[0]["name"] == "test_run")
check("accuracy correct", records[0]["accuracy"] == 0.5)

log_experiment("test_run_2", {"model": "test2"}, 0.6, 6, 10, "second run", experiments_file=tracker_path)
records = load_experiments(tracker_path)
check("logged 2 records", len(records) == 2)
os.unlink(tracker_path)

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)
