"""
Offline GRPO training for Qwen2.5-Coder-7B-Instruct on BIRD text-to-SQL.

Two-phase approach:
  Phase 1 (vLLM): Generate K completions per task, execute SQL, compute rewards
  Phase 2 (HF + peft): Train LoRA on fixed rollouts using GRPO loss

Usage:
    modal run rl/train.py                          # full run
    modal run rl/train.py --dry-run                # test with 10 tasks
    modal run rl/train.py --num-tasks 500          # smaller run
"""

import modal
import os
import time

app = modal.App("bird-climb-rl")

models_volume = modal.Volume.from_name("bird-climb-models", create_if_missing=True)
train_volume = modal.Volume.from_name("bird-climb-train-data")

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "vllm>=0.8.0",
        "huggingface_hub",
    )
)

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Inlined helpers (no imports from scaffold/ or eval/ — runs in Modal)

SYSTEM_PROMPT_V2 = """You are an expert SQLite SQL developer. Given a database schema and a natural language question, write a SQL query that answers the question.

Rules:
- Write SQLite-compatible SQL only
- Use backticks for column/table names with spaces or special characters
- Return ONLY the SQL query, no explanations or markdown
- Use the hint/evidence to understand domain-specific terminology
- SELECT only the columns the question asks for — no extra columns
- Only use DISTINCT when the question explicitly asks for unique/different/distinct values
- For division, always CAST the numerator to REAL to avoid integer division: CAST(x AS REAL) / y
- For date filtering, use strftime or SUBSTR on date columns as appropriate
- When the question asks "which one" or for a single value, return just that value, not a comparison table"""


def profile_database(db_path: str, sample_rows: int = 3, max_distinct: int = 20) -> dict:
    import sqlite3
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # Abort any query taking longer than 2 seconds
    conn.set_progress_handler(None, 0)  # reset first
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [row[0] for row in cursor.fetchall()]
    tables = []
    all_foreign_keys = []
    for table_name in table_names:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        row = cursor.fetchone()
        create_sql = row[0] if row else ""
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
        cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
        foreign_keys = []
        for fk in cursor.fetchall():
            fk_dict = {"from_table": table_name, "from_column": fk[3], "to_table": fk[2], "to_column": fk[4]}
            foreign_keys.append(fk_dict)
            all_foreign_keys.append(fk_dict)
        try:
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT ?", (sample_rows,))
            col_names = [desc[0] for desc in cursor.description]
            samples = [dict(zip(col_names, row)) for row in cursor.fetchall()]
        except Exception:
            samples = []
            col_names = [c["name"] for c in columns]
        try:
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            row_count = cursor.fetchone()[0]
        except Exception:
            row_count = 0
        column_values = {}
        if row_count <= 100000:  # skip distinct value scan for huge tables
            for col in columns:
                col_name = col["name"]
                try:
                    cursor.execute(f"SELECT COUNT(DISTINCT `{col_name}`) FROM `{table_name}`")
                    n_distinct = cursor.fetchone()[0]
                    if 0 < n_distinct <= max_distinct:
                        cursor.execute(f"SELECT DISTINCT `{col_name}` FROM `{table_name}` ORDER BY `{col_name}` LIMIT ?", (max_distinct,))
                        column_values[col_name] = [r[0] for r in cursor.fetchall()]
                except Exception:
                    continue
        tables.append({
            "name": table_name, "create_sql": create_sql, "columns": columns,
            "foreign_keys": foreign_keys, "sample_rows": samples,
            "column_names": col_names, "row_count": row_count, "column_values": column_values,
        })
    conn.close()
    return {"db_path": db_path, "db_name": os.path.basename(os.path.dirname(db_path)),
            "tables": tables, "foreign_keys": all_foreign_keys}


def format_profile(profile: dict) -> str:
    parts = [f"Database: {profile['db_name']}", ""]
    for table in profile["tables"]:
        parts.append(table["create_sql"] + ";")
        parts.append(f"-- {table['row_count']} rows")
        if table["sample_rows"]:
            parts.append(f"-- Sample rows from `{table['name']}`:")
            col_names = table["column_names"]
            for row in table["sample_rows"]:
                vals = [str(row.get(c, "NULL"))[:50] for c in col_names]
                parts.append(f"--   {', '.join(vals)}")
        if table["column_values"]:
            for col_name, values in table["column_values"].items():
                str_values = [str(v) if v is not None else "NULL" for v in values]
                vals_str = ", ".join(str_values[:15])
                if len(str_values) > 15:
                    vals_str += ", ..."
                parts.append(f"-- `{col_name}` distinct values: [{vals_str}]")
        parts.append("")
    if profile["foreign_keys"]:
        parts.append("-- Foreign Key Relationships:")
        for fk in profile["foreign_keys"]:
            parts.append(f"--   `{fk['from_table']}`.`{fk['from_column']}` -> `{fk['to_table']}`.`{fk['to_column']}`")
        parts.append("")
    return "\n".join(parts)


def build_prompt(schema_str: str, question: str, hint: str = "") -> str:
    parts = ["### Database Schema", schema_str.strip(), "", "### Question", question.strip()]
    if hint and hint.strip():
        parts.extend(["", "### Hint", hint.strip()])
    parts.extend(["", "### SQL"])
    return "\n".join(parts)


def extract_sql(text: str) -> str:
    import re
    match = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = text.strip().split("\n")
    sql_lines = []
    started = False
    for line in lines:
        stripped = line.strip().upper()
        if not started and stripped and any(
            stripped.startswith(kw) for kw in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]
        ):
            started = True
        if started:
            sql_lines.append(line)
    return "\n".join(sql_lines).strip() if sql_lines else text.strip()


def execute_sql_safe(db_path: str, sql: str, timeout: float = 5.0):
    """Execute SQL with real thread-based timeout."""
    import sqlite3
    import threading
    result_holder = [None, None]
    cancel = threading.Event()

    def _run():
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=wal;")
            conn.set_progress_handler(lambda: 1 if cancel.is_set() else 0, 10000)
            cursor = conn.cursor()
            cursor.execute(sql)
            result_holder[0] = cursor.fetchall()
            conn.close()
        except Exception as e:
            result_holder[1] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        cancel.set()
        thread.join(timeout=2.0)
        return None, f"Query timed out after {timeout}s"
    return result_holder[0], result_holder[1]


def compute_reward(generated_sql: str, gold_sql: str, db_path: str) -> float:
    def _normalize(results):
        def _nv(v):
            if v is None: return "NULL"
            if isinstance(v, (int, float)):
                try:
                    if float(v) == int(float(v)): return str(int(float(v)))
                except (ValueError, OverflowError): pass
                return str(v)
            return str(v).strip().lower()
        return sorted(tuple(_nv(v) for v in row) for row in results)

    gold_results, gold_err = execute_sql_safe(db_path, gold_sql)
    if gold_err: return 0.0
    pred_results, pred_err = execute_sql_safe(db_path, generated_sql)
    if pred_err: return 0.0
    if _normalize(pred_results) == _normalize(gold_results): return 1.0
    return 0.1


# Phase 1: Rollout collection (vLLM)

@app.function(
    image=base_image,
    gpu="B200",
    volumes={"/models": models_volume, "/train_data": train_volume},
    timeout=3600,
    scaledown_window=600,
)
def collect_rollouts(
    num_tasks: int = 2500,
    group_size: int = 4,
    max_prompt_tokens: int = 4096,
    max_gen_tokens: int = 512,
    dry_run: bool = False,
):
    import json
    from collections import Counter
    from vllm import LLM, SamplingParams

    if dry_run:
        num_tasks = 10
        print("=== DRY RUN: 10 tasks ===")

    print("=" * 60)
    print("Phase 1: Collecting rollouts with vLLM")
    print("=" * 60, flush=True)

    # Load tasks
    with open("/train_data/train.json") as f:
        all_tasks = json.load(f)

    db_dir = "/train_data/train_databases"
    db_counts = Counter(t["db_id"] for t in all_tasks)

    # Profile databases
    print(f"Profiling {len(db_counts)} databases...", flush=True)
    profile_cache = {}
    available_dbs = set()
    for i, db_id in enumerate(db_counts):
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if os.path.exists(db_path):
            try:
                profile_cache[db_id] = profile_database(db_path)
                available_dbs.add(db_id)
            except Exception as e:
                print(f"  Skip {db_id}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  Profiled {i+1}/{len(db_counts)} databases...", flush=True)
    print(f"Profiled {len(available_dbs)} databases", flush=True)

    tasks = [t for t in all_tasks if t["db_id"] in available_dbs]

    # Load model and tokenizer
    print("Loading vLLM model...", flush=True)
    llm = LLM(
        model=MODEL_ID,
        download_dir="/models",
        max_model_len=max_prompt_tokens + max_gen_tokens,
        trust_remote_code=True,
        dtype="auto",
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts and filter by length
    print("Building prompts...", flush=True)
    examples = []
    for task in tasks:
        db_id = task["db_id"]
        profile = profile_cache[db_id]
        schema_str = format_profile(profile)
        question = task["question"]
        hint = task.get("evidence", task.get("hint", ""))
        user_prompt = build_prompt(schema_str, question, hint)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if token_len > max_prompt_tokens:
            continue

        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        examples.append({
            "prompt_text": prompt_text,
            "gold_sql": task["SQL"],
            "db_path": db_path,
            "db_id": db_id,
        })

    # Subset
    if len(examples) > num_tasks:
        db_freq = Counter(ex["db_id"] for ex in examples)
        examples.sort(key=lambda x: -db_freq[x["db_id"]])
        examples = examples[:num_tasks]

    print(f"Generating for {len(examples)} tasks, group_size={group_size}", flush=True)

    # Generate all completions at once with vLLM
    prompt_texts = [ex["prompt_text"] for ex in examples]
    params = SamplingParams(
        max_tokens=max_gen_tokens,
        temperature=0.9,
        top_p=0.95,
        n=group_size,
    )

    t0 = time.time()
    outputs = llm.generate(prompt_texts, params)
    gen_time = time.time() - t0
    print(f"Generation done: {len(prompt_texts)} prompts × {group_size} = {len(prompt_texts)*group_size} completions in {gen_time:.1f}s", flush=True)

    # Execute and compute rewards
    print("Computing rewards...", flush=True)
    rollouts = []
    total_rewards = []
    t0 = time.time()

    for i, (example, output) in enumerate(zip(examples, outputs)):
        completions = []
        for gen_output in output.outputs:
            gen_text = gen_output.text.strip()
            sql = extract_sql(gen_text)
            reward = compute_reward(sql, example["gold_sql"], example["db_path"])
            completions.append({
                "text": gen_text,
                "sql": sql,
                "reward": reward,
            })
            total_rewards.append(reward)

        rollouts.append({
            "prompt_text": example["prompt_text"],
            "gold_sql": example["gold_sql"],
            "db_id": example["db_id"],
            "completions": completions,
        })

        if (i + 1) % 100 == 0:
            avg_r = sum(total_rewards) / len(total_rewards)
            exact = sum(1 for r in total_rewards if r == 1.0) / len(total_rewards)
            print(f"  [{i+1}/{len(examples)}] avg_reward={avg_r:.3f} exact_match={exact:.3f}", flush=True)

    reward_time = time.time() - t0
    avg_reward = sum(total_rewards) / len(total_rewards)
    exact_match = sum(1 for r in total_rewards if r == 1.0) / len(total_rewards)

    print(f"\nRollout stats:", flush=True)
    print(f"  Tasks: {len(rollouts)}", flush=True)
    print(f"  Completions: {len(total_rewards)}", flush=True)
    print(f"  Avg reward: {avg_reward:.4f}", flush=True)
    print(f"  Exact match: {exact_match:.4f}", flush=True)
    print(f"  Gen time: {gen_time:.1f}s, Reward time: {reward_time:.1f}s", flush=True)

    # Save rollouts
    rollout_path = "/models/rl_rollouts.json"
    with open(rollout_path, "w") as f:
        json.dump(rollouts, f)
    models_volume.commit()
    print(f"Saved rollouts to {rollout_path}", flush=True)

    return {"num_tasks": len(rollouts), "avg_reward": avg_reward, "exact_match": exact_match}


# Phase 2: Training (HF + peft)

@app.function(
    image=base_image,
    gpu="B200",
    volumes={"/models": models_volume},
    timeout=7200,
    scaledown_window=600,
)
def train_grpo(
    lr: float = 1e-5,
    kl_coeff: float = 0.05,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    log_every: int = 10,
    checkpoint_every: int = 200,
    dry_run: bool = False,
    epochs: int = 1,
):
    import json
    import random
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    print("=" * 60)
    print("Phase 2: GRPO Training")
    print("=" * 60, flush=True)

    # Load rollouts
    rollout_path = "/models/rl_rollouts.json"
    with open(rollout_path) as f:
        rollouts = json.load(f)
    print(f"Loaded {len(rollouts)} rollouts", flush=True)

    if dry_run:
        rollouts = rollouts[:10]
        checkpoint_every = 2
        print("=== DRY RUN: 10 rollouts ===", flush=True)

    # Filter to rollouts with reward variance (skip all-same groups)
    useful = []
    for r in rollouts:
        rewards = [c["reward"] for c in r["completions"]]
        if max(rewards) > min(rewards):
            useful.append(r)
    print(f"Rollouts with reward variance: {len(useful)}/{len(rollouts)}", flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir="/models")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model with LoRA
    print("Loading policy model + LoRA...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, cache_dir="/models",
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load reference model (frozen)
    print("Loading reference model (frozen)...", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, cache_dir="/models",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01,
    )
    steps_per_epoch = len(useful) // batch_size
    total_steps = steps_per_epoch * epochs
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=lr * 0.1)

    print(f"Training: {len(useful)} rollouts, batch_size={batch_size}, epochs={epochs}, steps={total_steps}", flush=True)

    # Training loop
    start_time = time.time()
    global_step = 0
    running_loss = 0.0
    running_kl = 0.0
    running_reward = 0.0
    running_count = 0

    CHECKPOINT_DIR = "/models/rl_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(epochs):
        random.shuffle(useful)
        print(f"\n--- Epoch {epoch+1}/{epochs} ---", flush=True)
        for batch_start in range(0, len(useful), batch_size):
            batch = useful[batch_start:batch_start + batch_size]
            if not batch:
                continue

            global_step += 1
            batch_loss = 0.0
            batch_kl = 0.0
            batch_rewards = []
            n_updates = 0

            for rollout in batch:
                prompt_text = rollout["prompt_text"]
                completions = rollout["completions"]
                rewards = [c["reward"] for c in completions]
                mean_r = sum(rewards) / len(rewards)
                std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
                advantages = [(r - mean_r) / std_r for r in rewards]

                batch_rewards.extend(rewards)

                for comp, advantage in zip(completions, advantages):
                    if abs(advantage) < 1e-6:
                        continue

                    full_text = prompt_text + comp["text"]
                    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
                    gen_ids = full_ids[len(prompt_ids):]

                    if len(gen_ids) == 0:
                        continue

                    full_tensor = torch.tensor([full_ids], device=model.device)
                    gen_tensor = torch.tensor(gen_ids, device=model.device)
                    prompt_len = len(prompt_ids)

                    model.train()
                    policy_logits = model(full_tensor).logits[:, prompt_len - 1:-1, :]
                    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                    token_log_probs = policy_log_probs.gather(
                        2, gen_tensor.unsqueeze(0).unsqueeze(-1)
                    ).squeeze(-1).squeeze(0)

                    with torch.no_grad():
                        ref_logits = ref_model(full_tensor).logits[:, prompt_len - 1:-1, :]
                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        ref_token_log_probs = ref_log_probs.gather(
                            2, gen_tensor.unsqueeze(0).unsqueeze(-1)
                        ).squeeze(-1).squeeze(0)

                    kl = (token_log_probs - ref_token_log_probs).mean()
                    adv_tensor = torch.tensor(advantage, device=model.device, dtype=torch.bfloat16)
                    loss = -(adv_tensor * token_log_probs.sum()) + kl_coeff * kl

                    total_comps = sum(len(r["completions"]) for r in batch)
                    (loss / total_comps).backward()

                    batch_loss += loss.item()
                    batch_kl += kl.item()
                    n_updates += 1

            if n_updates > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()

            running_loss += batch_loss / max(n_updates, 1)
            running_kl += batch_kl / max(n_updates, 1)
            running_reward += sum(batch_rewards)
            running_count += len(batch_rewards)

            if global_step % log_every == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / global_step) * (total_steps - global_step)
                avg_loss = running_loss / log_every
                avg_kl = running_kl / log_every
                avg_reward = running_reward / max(running_count, 1)
                exact = sum(1 for r in batch_rewards if r == 1.0) / max(len(batch_rewards), 1)
                print(
                    f"Step {global_step}/{total_steps} | "
                    f"loss={avg_loss:.4f} | kl={avg_kl:.4f} | "
                    f"reward={avg_reward:.3f} | exact={exact:.3f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"elapsed={elapsed:.0f}s | eta={eta:.0f}s",
                    flush=True,
                )
                running_loss = 0.0
                running_kl = 0.0
                running_reward = 0.0
                running_count = 0

            if global_step % checkpoint_every == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                models_volume.commit()
                print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    # Save final
    final_path = os.path.join(CHECKPOINT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    models_volume.commit()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete! {global_step} steps in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Final checkpoint: {final_path}")
    print(f"{'=' * 60}", flush=True)

    return {"steps": global_step, "elapsed": elapsed, "checkpoint": final_path}


# Entrypoint

@app.local_entrypoint()
def main(
    num_tasks: int = 2500,
    group_size: int = 8,
    dry_run: bool = False,
    train_only: bool = False,
):
    if not train_only:
        print("=== Phase 1: Rollout Collection ===")
        rollout_result = collect_rollouts.remote(
            num_tasks=num_tasks,
            group_size=group_size,
            dry_run=dry_run,
        )
        print(f"Rollout result: {rollout_result}")

    print("\n=== Phase 2: GRPO Training ===")
    train_result = train_grpo.remote(
        dry_run=dry_run,
        lr=2e-5,
        epochs=2,
    )
    print(f"Training result: {train_result}")
