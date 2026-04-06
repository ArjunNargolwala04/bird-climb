"""
Majority vote via self-consistency.

For each task, sample N completions at temperature>0, execute all against
the database, and pick the SQL whose result set appears most often.

Usage:
    python -m inference.vote --data_dir data/dev --output results/vote_32b.json
    python -m inference.vote --data_dir data/dev --output results/vote_32b.json --n_samples 5 --temperature 0.7
"""

import json
import argparse
import os
import time
import hashlib
from collections import Counter

import modal

from eval.harness import load_dev_tasks, get_db_path, execute_sql, normalize_result_set, run_evaluation
from eval.tracker import log_experiment
from scaffold.prompt import build_prompt_for_task
from inference.generate import extract_sql


def hash_result_set(results: list[tuple]) -> str:
    """Hash a normalized result set for voting."""
    normalized = normalize_result_set(results)
    return hashlib.md5(str(normalized).encode()).hexdigest()


def pick_winner(candidates: list[dict]) -> str:
    """
    Pick the best SQL from a list of candidates using execution-based voting.

    Each candidate is: {"sql": str, "results": list|None, "error": str|None}

    Strategy:
    1. Filter to candidates that executed successfully
    2. Hash each result set
    3. Pick the result set that appears most often (mode)
    4. Return the SQL from the earliest candidate with that result set
    5. If no candidates executed, return the first one
    """
    # Filter to successful executions
    valid = [(i, c) for i, c in enumerate(candidates) if c["results"] is not None]

    if not valid:
        # All failed — return first candidate
        return candidates[0]["sql"]

    # Hash result sets and vote
    votes = Counter()
    hash_to_candidates = {}
    for i, c in valid:
        h = hash_result_set(c["results"])
        votes[h] += 1
        if h not in hash_to_candidates:
            hash_to_candidates[h] = (i, c)

    # Pick mode (most common result set), break ties by earliest index
    best_hash = votes.most_common(1)[0][0]
    _, best_candidate = hash_to_candidates[best_hash]
    return best_candidate["sql"]


def run_vote(
    data_dir: str,
    output_path: str,
    n_samples: int = 5,
    temperature: float = 0.7,
    batch_size: int = 20,
    max_tokens: int = 1024,
    limit: int | None = None,
    prompt_version: str = "v1",
    use_schema_linking: bool = False,
):
    """Run majority vote inference over dev set."""
    tasks = load_dev_tasks(data_dir)
    if limit:
        tasks = tasks[:limit]

    print(f"Running vote inference: {len(tasks)} tasks, n={n_samples}, temp={temperature}",
          flush=True)

    Qwen32B = modal.Cls.from_name("bird-climb", "Qwen32B")
    model = Qwen32B()

    # Phase 1: Get all generations from Modal
    all_completions = {}  # task_idx -> list of raw completions
    total_time = 0

    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]

        prompts = []
        task_indices = []
        for task in batch_tasks:
            prompt = build_prompt_for_task(task, data_dir, prompt_version=prompt_version, use_schema_linking=use_schema_linking)
            prompts.append(prompt)
            task_indices.append(task["task_idx"])

        t0 = time.time()
        try:
            results = model.generate_batch.remote(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n_samples,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            for idx, completions in zip(task_indices, results):
                all_completions[idx] = completions

            print(
                f"  [inference] Batch {batch_start}-{batch_end}: {elapsed:.1f}s "
                f"({len(batch_tasks) / elapsed:.1f} tasks/s)",
                flush=True,
            )
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"  [inference] Batch {batch_start}-{batch_end}: ERROR ({e})", flush=True)
            for idx in task_indices:
                all_completions[idx] = ["SELECT 1"]

    print(f"\nInference done: {len(all_completions)} tasks in {total_time:.1f}s", flush=True)

    # Phase 2: Local SQL execution + voting
    print("Running local SQL execution and voting...", flush=True)
    predictions = {}
    vote_stats = {"unanimous": 0, "majority": 0, "no_majority": 0, "all_failed": 0}

    for i, task in enumerate(tasks):
        idx = task["task_idx"]
        if idx not in all_completions:
            continue

        db_path = get_db_path(data_dir, task["db_id"])
        candidates = []
        for raw in all_completions[idx]:
            sql = extract_sql(raw)
            exec_results, error = execute_sql(db_path, sql, timeout=5.0)
            candidates.append({"sql": sql, "results": exec_results, "error": error})

        winner = pick_winner(candidates)
        predictions[str(idx)] = winner

        valid = [c for c in candidates if c["results"] is not None]
        if not valid:
            vote_stats["all_failed"] += 1
        else:
            hashes = [hash_result_set(c["results"]) for c in valid]
            most_common_count = Counter(hashes).most_common(1)[0][1]
            if most_common_count == len(valid):
                vote_stats["unanimous"] += 1
            elif most_common_count > 1:
                vote_stats["majority"] += 1
            else:
                vote_stats["no_majority"] += 1

        if (i + 1) % 200 == 0:
            print(f"  [voting] {i+1}/{len(tasks)} tasks processed", flush=True)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nVote complete: {len(predictions)} predictions in {total_time:.1f}s")
    print(f"  Unanimous: {vote_stats['unanimous']}")
    print(f"  Majority:  {vote_stats['majority']}")
    print(f"  No majority (tie): {vote_stats['no_majority']}")
    print(f"  All failed: {vote_stats['all_failed']}")
    print(f"  Saved to {output_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Majority vote on BIRD dev set")
    parser.add_argument("--data_dir", type=str, default="data/dev")
    parser.add_argument("--output", type=str, default="results/vote_32b.json")
    parser.add_argument("--eval_output", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--prompt_version", type=str, default="v1")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--schema_linking", action="store_true",
                        help="Enable schema linking to filter irrelevant tables")

    args = parser.parse_args()

    predictions = run_vote(
        data_dir=args.data_dir,
        output_path=args.output,
        n_samples=args.n_samples,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        limit=args.limit,
        prompt_version=args.prompt_version,
        use_schema_linking=args.schema_linking,
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
        exp_name = args.name or f"vote_n{args.n_samples}"
        log_experiment(
            name=exp_name,
            config={
                "model": "Qwen2.5-Coder-32B-Instruct",
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "n_samples": args.n_samples,
                "prompt_version": args.prompt_version,
                "method": "majority_vote",
                "schema_linking": args.schema_linking,
            },
            accuracy=summary["execution_accuracy"],
            correct=summary["correct"],
            total=summary["total"],
            notes=f"vote n={args.n_samples} temp={args.temperature} prompt={args.prompt_version} sl={args.schema_linking}",
            results_path=eval_output,
        )


if __name__ == "__main__":
    main()
