"""
Self-repair: retry SQL that throws execution errors.

For any generated SQL that fails execution (syntax error, missing column, etc.),
make ONE retry call with the error message appended, asking the model to fix it.

Can be used standalone or integrated into the vote pipeline.

Usage:
    python -m inference.repair --data_dir data/dev --predictions results/baseline_32b.json --output results/repair_32b.json
"""

import json
import argparse
import os
import time

import modal

from eval.harness import load_dev_tasks, get_db_path, execute_sql, run_evaluation
from eval.tracker import log_experiment
from scaffold.prompt import build_prompt_for_task, SYSTEM_PROMPTS
from inference.generate import extract_sql


REPAIR_SUFFIX = """

### Error
The previous SQL query failed with the following error:
{error}

### Previous (broken) SQL
{broken_sql}

Fix the SQL query. Return ONLY the corrected SQL, nothing else."""


def repair_one(
    model,
    task: dict,
    broken_sql: str,
    error_msg: str,
    data_dir: str,
    max_tokens: int = 1024,
    prompt_version: str = "v1",
) -> str:
    """Attempt to repair a single broken SQL query."""
    # Build the original prompt and append error context
    prompt = build_prompt_for_task(task, data_dir, prompt_version=prompt_version)
    repair_context = REPAIR_SUFFIX.format(error=error_msg, broken_sql=broken_sql)
    prompt["user"] = prompt["user"] + repair_context

    results = model.generate.remote(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return extract_sql(results[0])


def run_repair(
    data_dir: str,
    predictions_path: str,
    output_path: str,
    max_tokens: int = 1024,
    prompt_version: str = "v1",
):
    """
    Load predictions, find ones that fail execution, attempt repair.
    Saves new predictions with repaired SQL where possible.
    """
    tasks = load_dev_tasks(data_dir)
    task_map = {t["task_idx"]: t for t in tasks}

    with open(predictions_path) as f:
        predictions = json.load(f)

    Qwen32B = modal.Cls.from_name("bird-climb", "Qwen32B")
    model = Qwen32B()

    repaired = dict(predictions)  # copy
    repair_count = 0
    repair_success = 0
    total_time = 0

    # Find broken predictions
    broken = []
    for idx_str, sql in predictions.items():
        idx = int(idx_str)
        if idx not in task_map:
            continue
        task = task_map[idx]
        db_path = get_db_path(data_dir, task["db_id"])
        _, error = execute_sql(db_path, sql)
        if error:
            broken.append((idx_str, sql, error, task))

    print(f"Found {len(broken)} broken predictions out of {len(predictions)}")

    for idx_str, old_sql, error_msg, task in broken:
        t0 = time.time()
        try:
            new_sql = repair_one(
                model, task, old_sql, error_msg, data_dir,
                max_tokens=max_tokens, prompt_version=prompt_version,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            # Check if repair actually fixed it
            db_path = get_db_path(data_dir, task["db_id"])
            _, new_error = execute_sql(db_path, new_sql)

            repair_count += 1
            if new_error is None:
                repaired[idx_str] = new_sql
                repair_success += 1
                print(f"  Task {idx_str}: FIXED ({elapsed:.1f}s)")
            else:
                print(f"  Task {idx_str}: still broken ({elapsed:.1f}s) — {new_error[:60]}")
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"  Task {idx_str}: repair call failed ({e})")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(repaired, f, indent=2)

    print(f"\nRepair complete in {total_time:.1f}s")
    print(f"  Attempted: {repair_count}")
    print(f"  Fixed: {repair_success}")
    print(f"  Still broken: {repair_count - repair_success}")
    print(f"  Saved to {output_path}")

    return repaired


def main():
    parser = argparse.ArgumentParser(description="Self-repair broken SQL predictions")
    parser.add_argument("--data_dir", type=str, default="data/dev")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON to repair")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save repaired predictions")
    parser.add_argument("--eval_output", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--prompt_version", type=str, default="v1")
    parser.add_argument("--name", type=str, default=None)

    args = parser.parse_args()

    repaired = run_repair(
        data_dir=args.data_dir,
        predictions_path=args.predictions,
        output_path=args.output,
        max_tokens=args.max_tokens,
        prompt_version=args.prompt_version,
    )

    if not args.skip_eval:
        eval_output = args.eval_output
        if not eval_output:
            base = args.output.replace(".json", "")
            eval_output = f"{base}_detailed.json"

        print(f"\nRunning evaluation...")
        result = run_evaluation(
            data_dir=args.data_dir,
            predictions_path=args.output,
            output_path=eval_output,
        )

        summary = result["summary"]
        exp_name = args.name or "repair"
        log_experiment(
            name=exp_name,
            config={
                "model": "Qwen2.5-Coder-32B-Instruct",
                "method": "self_repair",
                "prompt_version": args.prompt_version,
                "source": args.predictions,
            },
            accuracy=summary["execution_accuracy"],
            correct=summary["correct"],
            total=summary["total"],
            notes=f"repair on {os.path.basename(args.predictions)}",
            results_path=eval_output,
        )


if __name__ == "__main__":
    main()
