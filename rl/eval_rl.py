"""
Evaluate base 7B vs RL-trained 7B on BIRD dev set.

1. Deploys Qwen2.5-Coder-7B-Instruct on Modal
2. Runs baseline inference (greedy) → results/baseline_7b.json
3. Loads base + LoRA checkpoint, runs inference → results/rl_7b.json
4. Evaluates both and prints comparison

Usage:
    # First deploy the 7B model on Modal:
    modal deploy rl/eval_rl.py

    # Then run evaluation locally:
    python -m rl.eval_rl
    python -m rl.eval_rl --limit 100  # quick test
"""

import modal
import os
import re
import json
import time
import argparse

# Modal app for serving the 7B model

app = modal.App("bird-climb-7b")

models_volume = modal.Volume.from_name("bird-climb-models", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm>=0.8.0",
        "transformers",
        "torch",
        "peft",
    )
)


@app.cls(
    image=vllm_image,
    gpu="B200",
    volumes={"/models": models_volume},
    timeout=600,
    scaledown_window=300,
)
class Qwen7B:
    """Serves base Qwen2.5-Coder-7B-Instruct via vLLM."""

    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    @modal.enter()
    def load_model(self):
        from vllm import LLM

        self.llm = LLM(
            model=self.model_id,
            download_dir="/models",
            max_model_len=8192,
            trust_remote_code=True,
            dtype="auto",
        )
        self.tokenizer = self.llm.get_tokenizer()

    @modal.method()
    def generate_batch(
        self,
        prompts: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        n: int = 1,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        prompt_texts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ]
            prompt_texts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=[],
        )

        outputs = self.llm.generate(prompt_texts, params)
        return [
            [out.text.strip() for out in output.outputs]
            for output in outputs
        ]


@app.cls(
    image=vllm_image,
    gpu="B200",
    volumes={"/models": models_volume},
    timeout=600,
    scaledown_window=300,
)
class Qwen7BLoRA:
    """Serves Qwen2.5-Coder-7B-Instruct + LoRA checkpoint via vLLM."""

    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    lora_path: str = "/models/rl_checkpoints/final"

    @modal.enter()
    def load_model(self):
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.llm = LLM(
            model=self.model_id,
            download_dir="/models",
            max_model_len=8192,
            trust_remote_code=True,
            dtype="auto",
            enable_lora=True,
            max_lora_rank=32,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.lora_request = LoRARequest("rl_lora", 1, self.lora_path)

    @modal.method()
    def generate_batch(
        self,
        prompts: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        n: int = 1,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        prompt_texts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ]
            prompt_texts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=[],
        )

        outputs = self.llm.generate(
            prompt_texts, params, lora_request=self.lora_request
        )
        return [
            [out.text.strip() for out in output.outputs]
            for output in outputs
        ]


# Local evaluation logic

def extract_sql(text: str) -> str:
    """Extract SQL from model output."""
    match = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

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
        return "\n".join(sql_lines).strip()
    return text.strip()


def run_inference_with_model(
    model_cls_name: str,
    app_name: str,
    data_dir: str,
    output_path: str,
    batch_size: int = 50,
    limit: int | None = None,
):
    """Run inference using a deployed Modal model class."""
    # Import locally to avoid issues when just deploying
    from eval.harness import load_dev_tasks
    from scaffold.prompt import build_prompt_for_task

    tasks = load_dev_tasks(data_dir)
    if limit:
        tasks = tasks[:limit]

    print(f"Running inference with {model_cls_name} on {len(tasks)} tasks...")

    ModelCls = modal.Cls.from_name(app_name, model_cls_name)
    model = ModelCls()

    predictions = {}
    total_time = 0

    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]

        prompts = []
        task_indices = []
        for task in batch_tasks:
            prompt = build_prompt_for_task(task, data_dir, prompt_version="v2")
            prompts.append(prompt)
            task_indices.append(task["task_idx"])

        t0 = time.time()
        try:
            results = model.generate_batch.remote(
                prompts=prompts, max_tokens=1024, temperature=0.0,
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
            print(f"  Batch {batch_start}-{batch_end}: ERROR ({e})")
            for idx in task_indices:
                if str(idx) not in predictions:
                    predictions[str(idx)] = "SELECT 1"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Inference complete: {len(predictions)} predictions in {total_time:.1f}s")
    print(f"Saved to {output_path}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate base 7B vs RL-trained 7B")
    parser.add_argument("--data_dir", type=str, default="data/dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline if already run")
    parser.add_argument("--skip_rl", action="store_true", help="Skip RL model if already run")

    args = parser.parse_args()

    from eval.harness import run_evaluation
    from eval.tracker import log_experiment
    from eval.analyze import compare_runs

    # Baseline 7B 
    baseline_path = "results/baseline_7b.json"
    baseline_eval_path = "results/baseline_7b_detailed.json"

    if not args.skip_baseline:
        print("\n" + "=" * 60)
        print("STEP 1: Baseline Qwen2.5-Coder-7B-Instruct")
        print("=" * 60)
        run_inference_with_model(
            model_cls_name="Qwen7B",
            app_name="bird-climb-7b",
            data_dir=args.data_dir,
            output_path=baseline_path,
            batch_size=args.batch_size,
            limit=args.limit,
        )

        print("\nEvaluating baseline...")
        baseline_result = run_evaluation(
            args.data_dir, baseline_path, baseline_eval_path,
        )
        baseline_acc = baseline_result["summary"]["execution_accuracy"]

        log_experiment(
            name="baseline_7b",
            config={"model": "Qwen2.5-Coder-7B-Instruct", "temperature": 0.0, "prompt_version": "v2"},
            accuracy=baseline_acc,
            correct=baseline_result["summary"]["correct"],
            total=baseline_result["summary"]["total"],
            notes="7B greedy baseline, v2 prompt",
            results_path=baseline_eval_path,
        )
    else:
        print("Skipping baseline (--skip_baseline)")

    # RL-trained 7B 
    rl_path = "results/rl_7b.json"
    rl_eval_path = "results/rl_7b_detailed.json"

    if not args.skip_rl:
        print("\n" + "=" * 60)
        print("STEP 2: RL-trained Qwen2.5-Coder-7B-Instruct + LoRA")
        print("=" * 60)
        run_inference_with_model(
            model_cls_name="Qwen7BLoRA",
            app_name="bird-climb-7b",
            data_dir=args.data_dir,
            output_path=rl_path,
            batch_size=args.batch_size,
            limit=args.limit,
        )

        print("\nEvaluating RL model...")
        rl_result = run_evaluation(
            args.data_dir, rl_path, rl_eval_path,
        )
        rl_acc = rl_result["summary"]["execution_accuracy"]

        log_experiment(
            name="rl_7b",
            config={
                "model": "Qwen2.5-Coder-7B-Instruct",
                "temperature": 0.0,
                "prompt_version": "v2",
                "method": "GRPO+LoRA",
            },
            accuracy=rl_acc,
            correct=rl_result["summary"]["correct"],
            total=rl_result["summary"]["total"],
            notes="7B + GRPO LoRA, v2 prompt",
            results_path=rl_eval_path,
        )
    else:
        print("Skipping RL model (--skip_rl)")

    # Comparison 
    if os.path.exists(baseline_eval_path) and os.path.exists(rl_eval_path):
        print("\n" + "=" * 60)
        print("COMPARISON: Base 7B vs RL 7B")
        print("=" * 60)
        comp = compare_runs(baseline_eval_path, rl_eval_path)
        print(f"  Base 7B accuracy:  {comp['accuracy_a']:.4f}")
        print(f"  RL 7B accuracy:    {comp['accuracy_b']:.4f}")
        print(f"  Delta:             {comp['delta']:+.4f}")
        print(f"  Tasks gained:      {comp['gained']}")
        print(f"  Tasks lost:        {comp['lost']}")
        print(f"  Both correct:      {comp['both_right']}")
        print(f"  Both wrong:        {comp['both_wrong']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
