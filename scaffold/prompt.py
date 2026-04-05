"""
Prompt Construction for BIRD Text-to-SQL

Builds prompts that combine enriched database schema (from profile.py)
with the question and hint to produce a complete prompt for the LLM.

Usage:
    from scaffold.prompt import build_prompt, build_prompt_for_task
    
    prompt = build_prompt_for_task(task, data_dir="data/dev")
"""

import os
import json
from scaffold.profile import profile_database, format_profile


# Cache profiles so we don't re-profile the same database
_profile_cache: dict[str, dict] = {}


def get_cached_profile(db_path: str) -> dict:
    """Get or create a cached database profile."""
    if db_path not in _profile_cache:
        _profile_cache[db_path] = profile_database(db_path)
    return _profile_cache[db_path]


def clear_cache():
    """Clear the profile cache."""
    _profile_cache.clear()


SYSTEM_PROMPTS = {
    "v1": """You are an expert SQLite SQL developer. Given a database schema and a natural language question, write a SQL query that answers the question.

Rules:
- Write SQLite-compatible SQL only
- Use backticks for column/table names with spaces or special characters
- Return ONLY the SQL query, no explanations or markdown
- Use the hint/evidence to understand domain-specific terminology""",

    "v2": """You are an expert SQLite SQL developer. Given a database schema and a natural language question, write a SQL query that answers the question.

Rules:
- Write SQLite-compatible SQL only
- Use backticks for column/table names with spaces or special characters
- Return ONLY the SQL query, no explanations or markdown
- Use the hint/evidence to understand domain-specific terminology
- SELECT only the columns the question asks for — no extra columns
- Only use DISTINCT when the question explicitly asks for unique/different/distinct values
- For division, always CAST the numerator to REAL to avoid integer division: CAST(x AS REAL) / y
- For date filtering, use strftime or SUBSTR on date columns as appropriate
- When the question asks "which one" or for a single value, return just that value, not a comparison table""",
}

SYSTEM_PROMPT = SYSTEM_PROMPTS["v1"]


def build_prompt(
    schema_str: str,
    question: str,
    hint: str = "",
) -> str:
    """
    Build a complete prompt for text-to-SQL generation.
    
    Args:
        schema_str: Formatted database schema from format_profile()
        question: Natural language question
        hint: Evidence/hint string from BIRD
    
    Returns:
        Complete prompt string
    """
    parts = [
        "### Database Schema",
        schema_str.strip(),
        "",
        "### Question",
        question.strip(),
    ]
    
    if hint and hint.strip():
        parts.extend([
            "",
            "### Hint",
            hint.strip(),
        ])
    
    parts.extend([
        "",
        "### SQL",
    ])
    
    return "\n".join(parts)


def build_prompt_for_task(
    task: dict,
    data_dir: str,
    use_system_prompt: bool = True,
    prompt_version: str = "v1",
    use_schema_linking: bool = False,
) -> dict:
    """
    Build a complete prompt for a BIRD task.

    Args:
        task: Task dict from dev.json (must have db_id, question, evidence/hint)
        data_dir: Path to BIRD data directory (e.g. "data/dev")
        use_system_prompt: Whether to include system prompt
        prompt_version: Which system prompt variant to use
        use_schema_linking: If True, filter schema to relevant tables

    Returns:
        Dict with 'system' and 'user' prompt strings, ready for chat-format LLM
    """
    db_id = task["db_id"]
    db_path = os.path.join(data_dir, "dev_databases", db_id, f"{db_id}.sqlite")

    profile = get_cached_profile(db_path)

    question = task["question"]
    hint = task.get("evidence", task.get("hint", ""))

    if use_schema_linking:
        from scaffold.schema_link import link_schema
        linked = link_schema(profile, question, hint)
        schema_str = format_profile(linked)
    else:
        schema_str = format_profile(profile)

    user_prompt = build_prompt(schema_str, question, hint)

    result = {"user": user_prompt}
    if use_system_prompt:
        result["system"] = SYSTEM_PROMPTS.get(prompt_version, SYSTEM_PROMPTS["v1"])

    return result


def build_all_prompts(
    data_dir: str,
    tasks: list[dict] = None,
) -> list[dict]:
    """
    Build prompts for all tasks. Returns list of {task_idx, system, user, db_id}.
    
    Args:
        data_dir: Path to BIRD data directory
        tasks: Optional pre-loaded task list. If None, loads from data_dir/dev.json
    """
    if tasks is None:
        dev_json_path = os.path.join(data_dir, "dev.json")
        with open(dev_json_path, "r") as f:
            tasks = json.load(f)
        for i, task in enumerate(tasks):
            task["task_idx"] = i
    
    prompts = []
    for task in tasks:
        prompt = build_prompt_for_task(task, data_dir)
        prompts.append({
            "task_idx": task["task_idx"],
            "db_id": task["db_id"],
            "system": prompt.get("system", ""),
            "user": prompt["user"],
        })
    
    return prompts


# ── CLI for testing ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build prompts for BIRD tasks")
    parser.add_argument("--data_dir", type=str, default="data/dev",
                        help="Path to BIRD data directory")
    parser.add_argument("--task_idx", type=int, default=0,
                        help="Index of task to show prompt for")
    parser.add_argument("--count", action="store_true",
                        help="Count total prompt tokens (rough estimate)")
    
    args = parser.parse_args()
    
    dev_json_path = os.path.join(args.data_dir, "dev.json")
    with open(dev_json_path, "r") as f:
        tasks = json.load(f)
    for i, task in enumerate(tasks):
        task["task_idx"] = i
    
    task = tasks[args.task_idx]
    prompt = build_prompt_for_task(task, args.data_dir)
    
    print("=" * 60)
    print(f"Task {args.task_idx}: {task['question'][:80]}")
    print(f"DB: {task['db_id']}")
    print("=" * 60)
    print(f"\n[SYSTEM]\n{prompt['system']}")
    print(f"\n[USER]\n{prompt['user']}")
    print(f"\n[GOLD SQL]\n{task['SQL']}")
    
    if args.count:
        # Rough token estimate: ~4 chars per token
        all_prompts = build_all_prompts(args.data_dir, tasks)
        total_chars = sum(len(p["system"]) + len(p["user"]) for p in all_prompts)
        print(f"\nTotal prompt chars across all tasks: {total_chars:,}")
        print(f"Rough token estimate: ~{total_chars // 4:,}")