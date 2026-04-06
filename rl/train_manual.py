"""
GRPO training for Qwen2.5-Coder-7B-Instruct on BIRD text-to-SQL.

Runs entirely on Modal with GPU access. Uses a manual GRPO implementation
with transformers + peft (LoRA) + vLLM for fast generation.

Usage:
    modal run rl/train.py
    modal run rl/train.py --num-tasks 500 --epochs 1  # smaller test run
"""

import modal
import os
import time

app = modal.App("bird-climb-rl")

models_volume = modal.Volume.from_name("bird-climb-models", create_if_missing=True)
train_volume = modal.Volume.from_name("bird-climb-train-data")

training_image = (
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
        "bitsandbytes",
    )
)

# ---------------------------------------------------------------------------
# Inlined prompt construction (from scaffold/prompt.py and scaffold/profile.py)
# ---------------------------------------------------------------------------

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
    """Profile a SQLite database — inlined from scaffold/profile.py."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [row[0] for row in cursor.fetchall()]

    tables = []
    all_foreign_keys = []

    for table_name in table_names:
        # CREATE TABLE
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        row = cursor.fetchone()
        create_sql = row[0] if row else ""

        # Column info
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        columns = []
        for col in cursor.fetchall():
            columns.append({"name": col[1], "type": col[2], "notnull": bool(col[3]), "pk": bool(col[5])})

        # Foreign keys
        cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
        foreign_keys = []
        for fk in cursor.fetchall():
            fk_dict = {"from_table": table_name, "from_column": fk[3], "to_table": fk[2], "to_column": fk[4]}
            foreign_keys.append(fk_dict)
            all_foreign_keys.append(fk_dict)

        # Sample rows
        try:
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT ?", (sample_rows,))
            col_names = [desc[0] for desc in cursor.description]
            samples = [dict(zip(col_names, row)) for row in cursor.fetchall()]
        except Exception:
            samples = []
            col_names = [c["name"] for c in columns]

        # Row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            row_count = cursor.fetchone()[0]
        except Exception:
            row_count = 0

        # Distinct values for low-cardinality columns
        column_values = {}
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
    return {
        "db_path": db_path,
        "db_name": os.path.basename(os.path.dirname(db_path)),
        "tables": tables,
        "foreign_keys": all_foreign_keys,
    }


def format_profile(profile: dict) -> str:
    """Format database profile into prompt string — inlined from scaffold/profile.py."""
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
    """Build user prompt string."""
    parts = ["### Database Schema", schema_str.strip(), "", "### Question", question.strip()]
    if hint and hint.strip():
        parts.extend(["", "### Hint", hint.strip()])
    parts.extend(["", "### SQL"])
    return "\n".join(parts)


def extract_sql(text: str) -> str:
    """Extract SQL from model output."""
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
            stripped.startswith(kw)
            for kw in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "PRAGMA"]
        ):
            started = True
        if started:
            sql_lines.append(line)
    if sql_lines:
        return "\n".join(sql_lines).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Inlined reward function (from rl/reward.py)
# ---------------------------------------------------------------------------

def compute_reward(generated_sql: str, gold_sql: str, db_path: str, timeout: float = 5.0) -> float:
    """Compute execution-accuracy reward."""
    import sqlite3

    def _exec(sql):
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            conn.execute("PRAGMA journal_mode=wal;")
            cur = conn.cursor()
            cur.execute(sql)
            results = cur.fetchall()
            conn.close()
            return results, None
        except Exception as e:
            return None, str(e)

    def _normalize(results):
        def _nv(v):
            if v is None:
                return "NULL"
            if isinstance(v, (int, float)):
                try:
                    if float(v) == int(float(v)):
                        return str(int(float(v)))
                except (ValueError, OverflowError):
                    pass
                return str(v)
            return str(v).strip().lower()
        return sorted(tuple(_nv(v) for v in row) for row in results)

    gold_results, gold_err = _exec(gold_sql)
    if gold_err:
        return 0.0

    pred_results, pred_err = _exec(generated_sql)
    if pred_err:
        return 0.0

    if _normalize(pred_results) == _normalize(gold_results):
        return 1.0

    return 0.1  # partial credit: valid SQL, wrong result


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="B200",
    volumes={"/models": models_volume, "/train_data": train_volume},
    timeout=7200,
    scaledown_window=600,
)
def train(
    num_tasks: int = 2500,
    epochs: int = 1,
    batch_size: int = 4,
    group_size: int = 4,
    lr: float = 1e-5,
    kl_coeff: float = 0.05,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_prompt_tokens: int = 4096,
    max_gen_tokens: int = 512,
    checkpoint_every: int = 500,
    log_every: int = 10,
):
    import json
    import random
    import torch
    import torch.nn.functional as F
    from collections import Counter
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    print("=" * 60)
    print("GRPO Training: Qwen2.5-Coder-7B-Instruct on BIRD")
    print("=" * 60)

    MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
    CHECKPOINT_DIR = "/models/rl_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── 1. Load training data ────────────────────────────────────
    print("\n[1/6] Loading training data...")
    with open("/train_data/train.json") as f:
        all_tasks = json.load(f)
    for i, t in enumerate(all_tasks):
        t["task_idx"] = i

    # Count tasks per database
    db_counts = Counter(t["db_id"] for t in all_tasks)
    print(f"Total tasks: {len(all_tasks)}, databases: {len(db_counts)}")

    # ── 2. Profile databases ─────────────────────────────────────
    print("\n[2/6] Profiling training databases...")
    db_dir = "/train_data/train_databases"
    profile_cache = {}
    available_dbs = set()

    for db_id in db_counts:
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if os.path.exists(db_path):
            try:
                profile_cache[db_id] = profile_database(db_path)
                available_dbs.add(db_id)
            except Exception as e:
                print(f"  WARNING: Failed to profile {db_id}: {e}")
        else:
            print(f"  WARNING: Database not found: {db_path}")

    print(f"Successfully profiled {len(available_dbs)} databases")

    # Filter tasks to available databases
    tasks = [t for t in all_tasks if t["db_id"] in available_dbs]
    print(f"Tasks with available databases: {len(tasks)}")

    # ── 3. Build prompts and filter by length ────────────────────
    print("\n[3/6] Building prompts and filtering by token length...")

    # Load tokenizer for length filtering
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir="/models"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_examples = []
    for task in tasks:
        db_id = task["db_id"]
        profile = profile_cache[db_id]
        schema_str = format_profile(profile)
        question = task["question"]
        hint = task.get("evidence", task.get("hint", ""))
        user_prompt = build_prompt(schema_str, question, hint)

        # Build chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Filter by token length
        token_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if token_len > max_prompt_tokens:
            continue

        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        training_examples.append({
            "prompt_text": prompt_text,
            "gold_sql": task["SQL"],
            "db_path": db_path,
            "db_id": db_id,
            "task_idx": task["task_idx"],
            "token_len": token_len,
        })

    print(f"Tasks fitting in {max_prompt_tokens} tokens: {len(training_examples)}")

    # Subset to num_tasks, preferring databases with more tasks
    if len(training_examples) > num_tasks:
        # Sort by database frequency (most tasks first) for reward density
        db_freq = Counter(ex["db_id"] for ex in training_examples)
        training_examples.sort(key=lambda x: -db_freq[x["db_id"]])
        training_examples = training_examples[:num_tasks]
    print(f"Training on {len(training_examples)} tasks")

    # ── 4. Load models ───────────────────────────────────────────
    print("\n[4/6] Loading model and applying LoRA...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/models",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Reference model (frozen) for KL divergence
    print("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/models",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    total_steps = (len(training_examples) * epochs) // batch_size
    print(f"Total training steps: {total_steps}")

    # Cosine LR schedule
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)

    # ── 5. Training loop ─────────────────────────────────────────
    print("\n[5/6] Starting GRPO training...")
    start_time = time.time()
    global_step = 0
    total_rewards = []
    total_losses = []

    for epoch in range(epochs):
        random.shuffle(training_examples)
        epoch_rewards = []

        for batch_start in range(0, len(training_examples), batch_size):
            batch_end = min(batch_start + batch_size, len(training_examples))
            batch = training_examples[batch_start:batch_end]
            if len(batch) == 0:
                continue

            global_step += 1
            step_rewards = []
            step_log_ratios = []
            step_advantages = []

            # For each example in the batch, generate group_size completions
            for example in batch:
                prompt_text = example["prompt_text"]
                gold_sql = example["gold_sql"]
                db_path = example["db_path"]

                # Tokenize prompt
                prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)
                prompt_ids = prompt_ids.to(model.device)
                prompt_len = prompt_ids.shape[1]

                # Generate group_size completions
                model.eval()
                with torch.no_grad():
                    outputs = model.generate(
                        prompt_ids.expand(group_size, -1),
                        max_new_tokens=max_gen_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Compute rewards for each completion
                group_rewards = []
                group_texts = []
                for i in range(group_size):
                    gen_ids = outputs[i, prompt_len:]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    sql = extract_sql(gen_text)
                    reward = compute_reward(sql, gold_sql, db_path)
                    group_rewards.append(reward)
                    group_texts.append(gen_text)

                step_rewards.extend(group_rewards)
                mean_reward = sum(group_rewards) / len(group_rewards)

                # Compute advantages (reward - group mean)
                advantages = [(r - mean_reward) for r in group_rewards]

                # Skip if all rewards are the same (no signal)
                if all(a == 0.0 for a in advantages):
                    continue

                # Compute policy loss for each completion
                model.train()
                for i in range(group_size):
                    if advantages[i] == 0.0:
                        continue

                    gen_ids = outputs[i, prompt_len:]
                    full_ids = outputs[i].unsqueeze(0)

                    # Forward pass through policy model
                    policy_logits = model(full_ids).logits[:, prompt_len - 1:-1, :]
                    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                    token_log_probs = policy_log_probs.gather(2, gen_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

                    # Forward pass through reference model
                    with torch.no_grad():
                        ref_logits = ref_model(full_ids).logits[:, prompt_len - 1:-1, :]
                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        ref_token_log_probs = ref_log_probs.gather(2, gen_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

                    # KL divergence (per token, then mean)
                    kl = (token_log_probs - ref_token_log_probs).mean()

                    # GRPO loss: -advantage * log_prob + kl_coeff * kl
                    advantage = torch.tensor(advantages[i], device=model.device, dtype=torch.bfloat16)
                    policy_loss = -(advantage * token_log_probs.mean()) + kl_coeff * kl

                    step_log_ratios.append(kl.item())
                    step_advantages.append(advantages[i])

                    # Accumulate gradients
                    (policy_loss / (batch_size * group_size)).backward()

            # Update weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_rewards.extend(step_rewards)
            total_rewards.extend(step_rewards)

            # Logging
            if global_step % log_every == 0:
                elapsed = time.time() - start_time
                avg_reward = sum(step_rewards) / max(len(step_rewards), 1)
                avg_kl = sum(step_log_ratios) / max(len(step_log_ratios), 1) if step_log_ratios else 0.0
                reward_1_frac = sum(1 for r in step_rewards if r == 1.0) / max(len(step_rewards), 1)
                steps_remaining = total_steps - global_step
                eta = (elapsed / global_step) * steps_remaining if global_step > 0 else 0

                print(
                    f"Step {global_step}/{total_steps} | "
                    f"reward_mean={avg_reward:.3f} | "
                    f"reward_1_frac={reward_1_frac:.3f} | "
                    f"kl={avg_kl:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"elapsed={elapsed:.0f}s | "
                    f"eta={eta:.0f}s"
                )

            # Checkpoint
            if global_step % checkpoint_every == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                models_volume.commit()
                print(f"  Saved checkpoint: {ckpt_path}")

        # End of epoch stats
        epoch_avg = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        epoch_1_frac = sum(1 for r in epoch_rewards if r == 1.0) / max(len(epoch_rewards), 1)
        print(f"\n=== Epoch {epoch + 1}/{epochs} complete ===")
        print(f"  Avg reward: {epoch_avg:.4f}")
        print(f"  Exact match rate: {epoch_1_frac:.4f}")
        print(f"  Total steps: {global_step}")

    # ── 6. Save final checkpoint ─────────────────────────────────
    print("\n[6/6] Saving final checkpoint...")
    final_path = os.path.join(CHECKPOINT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    models_volume.commit()

    elapsed = time.time() - start_time
    overall_avg = sum(total_rewards) / max(len(total_rewards), 1)
    overall_1_frac = sum(1 for r in total_rewards if r == 1.0) / max(len(total_rewards), 1)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Total steps: {global_step}")
    print(f"  Overall avg reward: {overall_avg:.4f}")
    print(f"  Overall exact match: {overall_1_frac:.4f}")
    print(f"  Final checkpoint: {final_path}")
    print(f"{'=' * 60}")

    return {
        "total_steps": global_step,
        "avg_reward": overall_avg,
        "exact_match_rate": overall_1_frac,
        "elapsed_seconds": elapsed,
        "checkpoint_path": final_path,
    }


@app.local_entrypoint()
def main(
    num_tasks: int = 2500,
    epochs: int = 1,
    batch_size: int = 4,
    group_size: int = 4,
):
    result = train.remote(
        num_tasks=num_tasks,
        epochs=epochs,
        batch_size=batch_size,
        group_size=group_size,
    )
    print(f"\nResult: {result}")
