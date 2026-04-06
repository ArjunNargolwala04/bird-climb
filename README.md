# bird-climb

Maximizing execution accuracy on the [BIRD](https://bird-bench.github.io/) text-to-SQL benchmark using prompt engineering, inference-time scaling, and reinforcement learning.

## Results

| Method | Model | Accuracy | Notes |
|---|---|---|---|
| Greedy baseline | Qwen2.5-Coder-32B | 57.4% | v1 prompt |
| Improved prompt (v2) | Qwen2.5-Coder-32B | 59.2% | +1.8pp from prompt engineering |
| Majority vote (n=5) | Qwen2.5-Coder-32B | **61.0%** | Self-consistency, temp=0.7 |
| GPT-4o verifier (n=5) | Qwen2.5-Coder-32B + GPT-4o | 61.1% | GPT-4o picks best candidate |
| Majority vote + schema linking | Qwen2.5-Coder-32B | 60.0% | Schema linking slightly hurt |
| Greedy baseline | Qwen2.5-Coder-7B | 50.5% | v2 prompt |
| GRPO + LoRA | Qwen2.5-Coder-7B | **51.2%** | 2 epochs, LR 2e-5, 295 useful rollouts |

Evaluated on 1,534 BIRD dev tasks across 11 databases.

## Approach

### 1. Database Profiling (`scaffold/profile.py`)

Each SQLite database is profiled to extract rich schema information beyond raw CREATE TABLE statements:

- Column types, primary keys, NOT NULL constraints
- Foreign key relationships
- Sample rows (3 per table)
- Distinct value lists for low-cardinality columns (<=20 unique values)
- Row counts per table

This gives the model concrete examples of what the data looks like, reducing hallucinated column names and helping it understand domain-specific values.

### 2. Prompt Engineering (`scaffold/prompt.py`)

Two prompt versions were tested:

- **v1**: Basic instructions (write SQLite SQL, use backticks, return only SQL)
- **v2**: Adds rules for common failure modes — avoid extra columns, correct integer division with CAST, use DISTINCT only when asked, handle date filtering properly

v2 improved accuracy by +1.8pp over v1 on greedy decoding.

### 3. Schema Linking (`scaffold/schema_link.py`)

Keyword-based schema filtering that scores tables/columns by relevance to the question and hint. Uses token overlap, value matching, and foreign key traversal to select a subset of tables. This reduces prompt length but slightly hurt accuracy (-1pp), likely because it occasionally filters out tables needed for joins.

### 4. Inference-Time Scaling

Three inference strategies, all using the same v2 prompt:

- **Greedy** (`inference/generate.py`): Single deterministic completion per task
- **Majority vote** (`inference/vote.py`): Sample n=5 completions at temp=0.7, execute each against the database, vote on the most common result set. Best overall at 61.0%.
- **GPT-4o verification** (`inference/verify.py`): Sample n=5 candidates, execute them, then ask GPT-4o to pick the best based on the question and execution results. Marginal improvement over majority vote (+0.1pp) at much higher cost.

### 5. Reinforcement Learning (`rl/`)

Attempted GRPO (Group Relative Policy Optimization) on Qwen2.5-Coder-7B-Instruct using execution accuracy as the reward signal, inspired by [Snowflake's Arctic-Text2SQL-R1](https://arxiv.org/abs/2503.01346).

**Setup:**
- Two-phase offline approach: collect rollouts with vLLM, then train on fixed rollouts
- 1,000 training tasks from BIRD train set, group size 4 (4,000 completions total)
- LoRA rank 16, alpha 32, targeting all linear layers (40M trainable params)
- Reward: 1.0 for exact match, 0.1 for valid SQL with wrong results, 0.0 for errors

**RL iteration:**
- **Run 1** (LR 1e-5, 1 epoch, 74 steps): Flat (-0.1pp). Only 295/1000 rollouts had reward variance — too little gradient signal.
- **Run 2** (LR 5e-5, 3 epochs): Diverged. KL blew up to -0.65, loss exploded. LR too aggressive.
- **Run 3** (LR 2e-5, 2 epochs, 148 steps): **+0.72pp** (50.5% -> 51.2%). KL stayed controlled (-0.07), loss settled. Gained 73 tasks, lost 62, net +11.

**Key learnings:** LR sensitivity was critical — 2x too high diverged, 2x too low gave no signal. The base model's 70% exact match on training data meant most groups had zero advantage (all correct or all wrong). With more time: larger group size (8-16), higher temperature (0.9), online rollouts, and training on harder tasks would likely yield stronger gains.

## Infrastructure

All model inference and training runs on [Modal](https://modal.com) with NVIDIA B200 GPUs.

- `inference/modal_app.py`: vLLM serving for Qwen2.5-Coder-32B-Instruct
- `rl/eval_rl.py`: vLLM serving for 7B base and 7B+LoRA models
- `rl/train.py`: Two-phase training (rollout collection + GRPO optimization)
- `rl/setup_train_data.py`: Downloads 69 training databases to Modal Volume

Model weights and checkpoints stored in Modal Volume `bird-climb-models`. Training databases stored in `bird-climb-train-data`.

## Evaluation

Execution accuracy: execute predicted SQL and gold SQL against the database, compare result sets (order-insensitive, type-normalized).

```bash
# Run evaluation on predictions
python -m eval.harness --data_dir data/dev --predictions results/baseline_32b.json

# Error analysis
python -m eval.analyze results/baseline_32b_detailed.json

# Compare two runs
python -m eval.analyze results/baseline_32b_detailed.json --compare results/vote_v2_detailed.json
```

## Project Structure

```
scaffold/
  profile.py          # SQLite database profiler
  prompt.py           # Prompt construction (v1, v2 system prompts)
  schema_link.py      # Keyword-based schema linking

inference/
  modal_app.py        # Modal vLLM server (32B model)
  generate.py         # Batch greedy inference
  vote.py             # Majority vote (self-consistency)
  verify.py           # GPT-4o candidate verification

eval/
  harness.py          # Execution accuracy evaluation
  analyze.py          # Error categorization and run comparison
  tracker.py          # Experiment JSONL logger
  test_harness.py     # Smoke tests

rl/
  setup_train_data.py # Download train databases to Modal
  reward.py           # Execution-based reward function
  train.py            # GRPO training (two-phase: rollouts + optimization)
  train_manual.py     # Fallback: single-phase manual GRPO
  eval_rl.py          # 7B baseline vs RL evaluation + Modal serving
  dataset.py          # (unused)

data/
  dev/                # 1,534 dev tasks + 11 SQLite databases
  train/              # 6,601 filtered train tasks (databases on Modal)

results/              # Predictions, detailed eval JSONs, run logs
experiments.jsonl     # Experiment log (all runs with configs and accuracy)
```

## Reproducing

```bash
# Deploy the 32B model
modal deploy inference/modal_app.py

# Run greedy baseline
python -m inference.generate --data_dir data/dev --output results/baseline_32b.json --prompt_version v2

# Run majority vote
python -m inference.vote --data_dir data/dev --output results/vote_v2.json --prompt_version v2 --n_samples 5

# Run RL training
modal run rl/setup_train_data.py      # download train databases
modal run rl/train.py                 # train GRPO + LoRA

# Evaluate RL model
modal deploy rl/eval_rl.py
python -m rl.eval_rl
```
