"""
Error Analysis for BIRD Evaluation Results

Categorizes errors and provides summary statistics to guide
where to focus improvement efforts.

Usage:
    python -m eval.analyze results/baseline_32b.json
    python -m eval.analyze results/baseline_32b.json --compare results/vote_32b.json
"""

import json
import argparse
import re
from collections import Counter


def categorize_error(result: dict) -> str:
    """
    Categorize a failed prediction into an error type.
    
    Categories:
        - correct: result sets matched
        - syntax_error: SQL syntax was invalid
        - no_such_table: referenced a table that doesn't exist
        - no_such_column: referenced a column that doesn't exist  
        - wrong_answer: valid SQL but wrong result set
        - timeout: query took too long
        - other_error: some other execution error
        - gold_error: gold SQL itself failed (dataset issue)
    """
    if result.get("skipped"):
        return "gold_error"
    
    if result["match"]:
        return "correct"
    
    error = result.get("pred_error", "")
    if error:
        error_lower = error.lower()
        if "syntax" in error_lower or "near" in error_lower:
            return "syntax_error"
        if "no such table" in error_lower:
            return "no_such_table"
        if "no such column" in error_lower:
            return "no_such_column"
        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        if "ambiguous column" in error_lower:
            return "ambiguous_column"
        return "other_error"
    
    return "wrong_answer"


def analyze_results(results_path: str) -> dict:
    """Analyze a detailed results file from the eval harness."""
    with open(results_path, "r") as f:
        data = json.load(f)
    
    results = data["results"]
    summary = data["summary"]
    
    # Categorize all errors
    categories = Counter()
    difficulty_breakdown = {}
    db_errors = {}
    examples = {}  # store a few examples per category
    
    for r in results:
        cat = categorize_error(r)
        categories[cat] += 1
        
        # Track by difficulty if available
        difficulty = r.get("difficulty", "unknown")
        if difficulty not in difficulty_breakdown:
            difficulty_breakdown[difficulty] = Counter()
        difficulty_breakdown[difficulty][cat] += 1
        
        # Track by database
        db_id = r["db_id"]
        if db_id not in db_errors:
            db_errors[db_id] = Counter()
        db_errors[db_id][cat] += 1
        
        # Store up to 3 examples per error category
        if cat != "correct" and cat not in examples:
            examples[cat] = []
        if cat != "correct" and len(examples[cat]) < 3:
            examples[cat].append({
                "task_idx": r["task_idx"],
                "db_id": r["db_id"],
                "question": r["question"],
                "gold_sql": r["gold_sql"],
                "predicted_sql": r["predicted_sql"],
                "error": r.get("pred_error", "result mismatch"),
            })
    
    return {
        "summary": summary,
        "error_categories": dict(categories.most_common()),
        "difficulty_breakdown": {
            k: dict(v.most_common()) for k, v in difficulty_breakdown.items()
        },
        "db_errors": {
            k: dict(v.most_common()) for k, v in db_errors.items()
        },
        "examples": examples,
    }


def compare_runs(path_a: str, path_b: str) -> dict:
    """Compare two evaluation runs to see which tasks flipped."""
    with open(path_a, "r") as f:
        data_a = json.load(f)
    with open(path_b, "r") as f:
        data_b = json.load(f)
    
    results_a = {r["task_idx"]: r for r in data_a["results"]}
    results_b = {r["task_idx"]: r for r in data_b["results"]}
    
    common_tasks = set(results_a.keys()) & set(results_b.keys())
    
    gained = []   # wrong in A, correct in B
    lost = []     # correct in A, wrong in B
    both_wrong = []
    both_right = []
    
    for idx in sorted(common_tasks):
        a_match = results_a[idx].get("match", False)
        b_match = results_b[idx].get("match", False)
        
        if not a_match and b_match:
            gained.append(idx)
        elif a_match and not b_match:
            lost.append(idx)
        elif a_match and b_match:
            both_right.append(idx)
        else:
            both_wrong.append(idx)
    
    return {
        "accuracy_a": data_a["summary"]["execution_accuracy"],
        "accuracy_b": data_b["summary"]["execution_accuracy"],
        "delta": round(data_b["summary"]["execution_accuracy"] - data_a["summary"]["execution_accuracy"], 4),
        "gained": len(gained),
        "lost": len(lost),
        "both_right": len(both_right),
        "both_wrong": len(both_wrong),
        "gained_task_ids": gained[:20],  # show first 20
        "lost_task_ids": lost[:20],
    }


def print_analysis(analysis: dict):
    """Pretty-print analysis results."""
    print(f"\n{'='*60}")
    print(f"BIRD Error Analysis")
    print(f"{'='*60}")
    print(f"Execution Accuracy: {analysis['summary']['execution_accuracy']:.4f}")
    print(f"Total tasks: {analysis['summary']['total']}")
    
    print(f"\nError Categories:")
    for cat, count in analysis["error_categories"].items():
        pct = count / max(analysis["summary"]["total"], 1) * 100
        print(f"  {cat:20s} {count:5d} ({pct:5.1f}%)")
    
    print(f"\nExample errors:")
    for cat, exs in analysis["examples"].items():
        print(f"\n  [{cat}]")
        for ex in exs[:2]:
            print(f"    Task {ex['task_idx']} ({ex['db_id']})")
            print(f"    Q: {ex['question'][:80]}...")
            print(f"    Gold: {ex['gold_sql'][:80]}...")
            print(f"    Pred: {ex['predicted_sql'][:80]}...")
            print(f"    Error: {ex['error'][:80]}")
    
    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Analyze BIRD evaluation results")
    parser.add_argument("results", type=str, help="Path to detailed results JSON")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to second results JSON for comparison")
    
    args = parser.parse_args()
    
    analysis = analyze_results(args.results)
    print_analysis(analysis)
    
    if args.compare:
        comp = compare_runs(args.results, args.compare)
        print(f"\nComparison: A={comp['accuracy_a']:.4f} -> B={comp['accuracy_b']:.4f} (delta={comp['delta']:+.4f})")
        print(f"  Gained: {comp['gained']} tasks")
        print(f"  Lost:   {comp['lost']} tasks")
        print(f"  Both right: {comp['both_right']}")
        print(f"  Both wrong: {comp['both_wrong']}")


if __name__ == "__main__":
    main()