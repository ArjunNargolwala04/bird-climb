"""
Batch inference over BIRD dev set using Modal-hosted Qwen2.5-Coder-32B.

Usage:
    python -m inference.generate --data_dir data/dev --output results/baseline_32b.json
    python -m inference.generate --data_dir data/dev --output results/baseline_32b.json --batch_size 50
"""

import json
import argparse
import os
import re
import time
import sys

import modal

from eval.harness import load_dev_tasks, run_evaluation
from eval.tracker import log_experiment
from scaffold.prompt import build_prompt_for_task


def extract_sql(text: str) -> str:
    """
    Extract SQL from model output. Handles:
    - Raw SQL
    - ```sql ... ``` blocks
    - Leading/trailing commentary
    """
    # Try to extract from markdown code block
    match = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Strip leading commentary lines (lines not starting with SELECT/WITH/INSERT/etc.)
    lines = text.strip().split("\n")
    sql_lines = []
    started = False
    for line in lines:
        stripped = line.strip().upper()
        if not started and stripped and any(
            stripped.startswith(kw)
            for kw in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "PRAGMA"]
        ):
            started = True
        if started:
            sql_lines.append(line)

    if sql_lines:
        return "\n".join(sql_lines).strip().rstrip(";") + ""

    # Fallback: return as-is
    return text.strip()


def run_inference(
    data_dir: str,
    output_path: str,
    batch_size: int = 50,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    limit: int | None = None,
    prompt_version: str = "v1",
    use_schema_linking: bool = False,
):
    """Run batched inference over dev set."""
    tasks = load_dev_tasks(data_dir)
    if limit:
        tasks = tasks[:limit]

    print(f"Running inference on {len(tasks)} tasks (batch_size={batch_size})")

    # Look up the remote Modal class
    Qwen32B = modal.Cls.from_name("bird-climb", "Qwen32B")
    model = Qwen32B()

    predictions = {}
    total_time = 0
    errors = 0

    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]

        # Build prompts
        prompts = []
        task_indices = []
        for task in batch_tasks:
            prompt = build_prompt_for_task(task, data_dir, prompt_version=prompt_version, use_schema_linking=use_schema_linking)
            prompts.append(prompt)
            task_indices.append(task["task_idx"])

        # Call Modal
        t0 = time.time()
        try:
            results = model.generate_batch.remote(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            for idx, completions in zip(task_indices, results):
                sql = extract_sql(completions[0])
                predictions[str(idx)] = sql

            print(
                f"  Batch {batch_start}-{batch_end}: {elapsed:.1f}s "
                f"({len(batch_tasks) / elapsed:.1f} tasks/s)"
            )
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            errors += 1
            print(f"  Batch {batch_start}-{batch_end}: ERROR ({e})")
            # Fill with empty predictions so eval still works
            for idx in task_indices:
                if str(idx) not in predictions:
                    predictions[str(idx)] = "SELECT 1"

    # Save predictions
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nInference complete: {len(predictions)} predictions in {total_time:.1f}s")
    if errors:
        print(f"  {errors} batch errors (filled with SELECT 1)")
    print(f"  Saved to {output_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Batch inference on BIRD dev set")
    parser.add_argument("--data_dir", type=str, default="data/dev")
    parser.add_argument("--output", type=str, default="results/baseline_32b.json")
    parser.add_argument("--eval_output", type=str, default=None,
                        help="Path for detailed eval results (default: auto from --output)")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run on first N tasks (for testing)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation after inference")
    parser.add_argument("--prompt_version", type=str, default="v1",
                        help="Prompt version to use (v1, v2, etc.)")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name for tracker (default: auto)")
    parser.add_argument("--schema_linking", action="store_true",
                        help="Enable schema linking to filter irrelevant tables")

    args = parser.parse_args()

    # Run inference
    predictions = run_inference(
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        limit=args.limit,
        prompt_version=args.prompt_version,
        use_schema_linking=args.schema_linking,
    )

    # Run evaluation
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

        # Log experiment
        summary = result["summary"]
        exp_name = args.name or f"prompt_{args.prompt_version}"
        log_experiment(
            name=exp_name,
            config={
                "model": "Qwen2.5-Coder-32B-Instruct",
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "n_samples": 1,
                "prompt_version": args.prompt_version,
            },
            accuracy=summary["execution_accuracy"],
            correct=summary["correct"],
            total=summary["total"],
            notes=f"prompt={args.prompt_version}, {len(predictions)} tasks",
            results_path=eval_output,
        )


if __name__ == "__main__":
    main()
