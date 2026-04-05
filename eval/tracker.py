"""
Lightweight experiment tracker.
Appends one JSON line per run to experiments.jsonl.

Usage:
    from eval.tracker import log_experiment
    
    log_experiment(
        name="baseline_32b",
        config={"model": "Qwen2.5-Coder-32B", "temperature": 0, "n_samples": 1},
        accuracy=0.412,
        notes="First baseline with basic schema prompt"
    )
"""

import json
import os
from datetime import datetime, timezone


EXPERIMENTS_FILE = "experiments.jsonl"


def log_experiment(
    name: str,
    config: dict,
    accuracy: float,
    correct: int = 0,
    total: int = 0,
    notes: str = "",
    results_path: str = "",
    experiments_file: str = EXPERIMENTS_FILE,
):
    """Append an experiment record to the JSONL log."""
    record = {
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "config": config,
        "notes": notes,
        "results_path": results_path,
    }
    
    with open(experiments_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    
    print(f"[tracker] Logged: {name} | accuracy={accuracy:.4f} | {notes}")
    return record


def load_experiments(experiments_file: str = EXPERIMENTS_FILE) -> list[dict]:
    """Load all experiment records."""
    if not os.path.exists(experiments_file):
        return []
    
    records = []
    with open(experiments_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_experiments(experiments_file: str = EXPERIMENTS_FILE):
    """Print a summary table of all experiments."""
    records = load_experiments(experiments_file)
    
    if not records:
        print("No experiments logged yet.")
        return
    
    print(f"\n{'Name':<25} {'Accuracy':>8} {'Correct':>8} {'Total':>6} {'Notes'}")
    print("-" * 80)
    for r in records:
        print(f"{r['name']:<25} {r['accuracy']:>8.4f} {r['correct']:>8} {r['total']:>6} {r.get('notes', '')[:30]}")
    print()


if __name__ == "__main__":
    print_experiments()