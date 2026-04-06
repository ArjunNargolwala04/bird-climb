"""
GPT-4 Verifier: Generate N candidates with Qwen, have GPT-4 pick the best.

Usage:
    python -u -m inference.verify --data_dir data/dev --output results/verify_v2.json --prompt_version v2
"""

import json
import argparse
import os
import time
import asyncio
from collections import Counter

import modal
from openai import AsyncOpenAI

from eval.harness import load_dev_tasks, get_db_path, execute_sql, normalize_result_set, run_evaluation
from eval.tracker import log_experiment
from scaffold.prompt import build_prompt_for_task
from inference.generate import extract_sql


VERIFY_PROMPT = """You are an expert SQL judge. Given a question about a database, several candidate SQL queries, and their execution results, pick the BEST candidate.

### Question
{question}

### Hint
{hint}

### Candidates
{candidates_text}

Rules:
- Pick the candidate whose result most correctly answers the question
- Prefer candidates that return non-empty results over empty ones
- Prefer candidates that return the right number of columns/rows for the question
- If the question asks for a single value, prefer candidates returning one row
- If the question asks for a list, prefer candidates returning multiple rows

Reply with ONLY the candidate number (e.g. "1" or "3"). Nothing else."""


def format_candidates_for_verify(candidates: list[dict]) -> str:
    """Format candidates with their execution results for GPT-4."""
    parts = []
    for i, c in enumerate(candidates, 1):
        parts.append(f"Candidate {i}:")
        parts.append(f"  SQL: {c['sql']}")
        if c["error"]:
            parts.append(f"  Result: ERROR - {c['error'][:100]}")
        elif c["results"] is not None:
            n_rows = len(c["results"])
            n_cols = len(c["results"][0]) if c["results"] else 0
            preview = str(c["results"][:5])[:200]
            parts.append(f"  Result: {n_rows} rows, {n_cols} cols — {preview}")
        parts.append("")
    return "\n".join(parts)


async def verify_batch_async(
    client: AsyncOpenAI,
    batch: list[dict],
    model: str = "gpt-4o",
    max_concurrent: int = 20,
) -> list[int]:
    """Call GPT-4 to verify a batch of tasks. Returns list of chosen indices (0-based)."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def verify_one(item):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": item["prompt"]}],
                    max_tokens=5,
                    temperature=0,
                )
                choice = response.choices[0].message.content.strip()
                # Parse the number
                num = int("".join(c for c in choice if c.isdigit()))
                return max(0, num - 1)  # Convert 1-indexed to 0-indexed
            except Exception as e:
                return 0  # Default to first candidate on error

    results = await asyncio.gather(*[verify_one(item) for item in batch])
    return results


def run_verify(
    data_dir: str,
    output_path: str,
    n_samples: int = 5,
    temperature: float = 0.7,
    batch_size: int = 20,
    max_tokens: int = 1024,
    limit: int | None = None,
    prompt_version: str = "v1",
    verify_model: str = "gpt-4o",
    openai_api_key: str | None = None,
):
    """Generate candidates with Qwen, verify with GPT-4."""
    tasks = load_dev_tasks(data_dir)
    if limit:
        tasks = tasks[:limit]

    print(f"Running verify pipeline: {len(tasks)} tasks, n={n_samples}, verifier={verify_model}",
          flush=True)

    # Phase 1: Generate candidates with Qwen on Modal
    Qwen32B = modal.Cls.from_name("bird-climb", "Qwen32B")
    model = Qwen32B()

    all_completions = {}
    total_inference_time = 0

    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]

        prompts = []
        task_indices = []
        for task in batch_tasks:
            prompt = build_prompt_for_task(task, data_dir, prompt_version=prompt_version)
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
            total_inference_time += elapsed

            for idx, completions in zip(task_indices, results):
                all_completions[idx] = completions

            print(f"  [inference] Batch {batch_start}-{batch_end}: {elapsed:.1f}s", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            total_inference_time += elapsed
            print(f"  [inference] Batch {batch_start}-{batch_end}: ERROR ({e})", flush=True)
            for idx in task_indices:
                all_completions[idx] = ["SELECT 1"]

    print(f"\nInference done: {len(all_completions)} tasks in {total_inference_time:.1f}s", flush=True)

    # Phase 2: Execute all candidates locally
    print("Executing candidates locally...", flush=True)
    all_candidates = {}  # task_idx -> list of candidate dicts

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
        all_candidates[idx] = candidates

        if (i + 1) % 200 == 0:
            print(f"  [execute] {i+1}/{len(tasks)} tasks", flush=True)

    print(f"Execution done.", flush=True)

    # Phase 3: For tasks where candidates disagree, ask GPT-4 to pick
    print("Running GPT-4 verification...", flush=True)

    client = AsyncOpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))
    predictions = {}
    verify_items = []
    verify_task_indices = []
    unanimous_count = 0
    verified_count = 0

    for task in tasks:
        idx = task["task_idx"]
        candidates = all_candidates.get(idx, [])
        if not candidates:
            predictions[str(idx)] = "SELECT 1"
            continue

        # Check if all valid candidates agree (unanimous)
        valid = [c for c in candidates if c["results"] is not None]
        if valid:
            from inference.vote import hash_result_set
            hashes = [hash_result_set(c["results"]) for c in valid]
            counts = Counter(hashes)
            if len(counts) == 1:
                # All agree — no need to verify
                predictions[str(idx)] = valid[0]["sql"]
                unanimous_count += 1
                continue

        # Candidates disagree — ask GPT-4
        candidates_text = format_candidates_for_verify(candidates)
        verify_prompt = VERIFY_PROMPT.format(
            question=task["question"],
            hint=task.get("hint", task.get("evidence", "")),
            candidates_text=candidates_text,
        )
        verify_items.append({"prompt": verify_prompt, "candidates": candidates})
        verify_task_indices.append(idx)

    print(f"  Unanimous (skip verify): {unanimous_count}", flush=True)
    print(f"  Need verification: {len(verify_items)}", flush=True)

    # Run GPT-4 verification in batches
    async def run_all_verify():
        batch_sz = 50
        all_choices = []
        for start in range(0, len(verify_items), batch_sz):
            end = min(start + batch_sz, len(verify_items))
            batch = verify_items[start:end]
            choices = await verify_batch_async(client, batch, model=verify_model)
            all_choices.extend(choices)
            print(f"  [verify] {end}/{len(verify_items)} tasks verified", flush=True)
        return all_choices

    choices = asyncio.run(run_all_verify())

    for idx, choice in zip(verify_task_indices, choices):
        candidates = all_candidates[idx]
        chosen_idx = min(choice, len(candidates) - 1)
        predictions[str(idx)] = candidates[chosen_idx]["sql"]
        verified_count += 1

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nVerify complete: {len(predictions)} predictions", flush=True)
    print(f"  Unanimous (majority vote): {unanimous_count}", flush=True)
    print(f"  GPT-4 verified: {verified_count}", flush=True)
    print(f"  Saved to {output_path}", flush=True)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="GPT-4 verified inference on BIRD dev set")
    parser.add_argument("--data_dir", type=str, default="data/dev")
    parser.add_argument("--output", type=str, default="results/verify_v2.json")
    parser.add_argument("--eval_output", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--prompt_version", type=str, default="v2")
    parser.add_argument("--verify_model", type=str, default="gpt-4o")
    parser.add_argument("--name", type=str, default=None)

    args = parser.parse_args()

    predictions = run_verify(
        data_dir=args.data_dir,
        output_path=args.output,
        n_samples=args.n_samples,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        limit=args.limit,
        prompt_version=args.prompt_version,
        verify_model=args.verify_model,
    )

    if not args.skip_eval:
        eval_output = args.eval_output
        if not eval_output:
            base = args.output.replace(".json", "")
            eval_output = f"{base}_detailed.json"

        print(f"\nRunning evaluation...", flush=True)
        result = run_evaluation(
            data_dir=args.data_dir,
            predictions_path=args.output,
            output_path=eval_output,
        )

        summary = result["summary"]
        exp_name = args.name or f"verify_{args.verify_model}"
        log_experiment(
            name=exp_name,
            config={
                "model": "Qwen2.5-Coder-32B-Instruct",
                "temperature": args.temperature,
                "n_samples": args.n_samples,
                "prompt_version": args.prompt_version,
                "verifier": args.verify_model,
                "method": "gpt4_verify",
            },
            accuracy=summary["execution_accuracy"],
            correct=summary["correct"],
            total=summary["total"],
            notes=f"verify {args.verify_model} n={args.n_samples} prompt={args.prompt_version}",
            results_path=eval_output,
        )


if __name__ == "__main__":
    main()
