"""
Schema Linking for BIRD Text-to-SQL

Given a question and hint, identify which tables and columns are most
relevant. This filters the schema to reduce noise and help the model
focus on the right tables.

Uses keyword matching + fuzzy matching between question/hint terms
and table/column names. No ML model needed.

Usage:
    from scaffold.schema_link import link_schema
    relevant = link_schema(profile, question, hint)
"""

import re
import sqlite3
import os
from scaffold.profile import profile_database, format_profile


def tokenize(text: str) -> set[str]:
    """Split text into lowercase tokens, removing punctuation."""
    return set(re.findall(r'[a-z][a-z0-9_]*', text.lower()))


def normalize_name(name: str) -> set[str]:
    """
    Break a column/table name into searchable tokens.
    'Free Meal Count (K-12)' -> {'free', 'meal', 'count', 'k', '12'}
    'CDSCode' -> {'cds', 'code', 'cdscode'}
    """
    # Split on spaces, underscores, camelCase, parens
    parts = re.findall(r'[a-z]+|[A-Z][a-z]*|[0-9]+', name)
    tokens = {p.lower() for p in parts}
    # Also add the full lowered name
    tokens.add(re.sub(r'[^a-z0-9]', '', name.lower()))
    return tokens


def score_table(table: dict, question_tokens: set[str], hint_tokens: set[str]) -> float:
    """
    Score a table's relevance to the question + hint.
    Higher = more relevant.
    """
    score = 0.0
    all_query_tokens = question_tokens | hint_tokens

    # Table name match
    table_tokens = normalize_name(table["name"])
    table_overlap = table_tokens & all_query_tokens
    score += len(table_overlap) * 3.0

    # Column name matches
    for col in table["columns"]:
        col_tokens = normalize_name(col["name"])
        col_overlap = col_tokens & all_query_tokens
        if col_overlap:
            score += len(col_overlap) * 2.0

    # Value matches — check if question mentions values in low-cardinality columns
    for col_name, values in table.get("column_values", {}).items():
        for val in values:
            if val is None:
                continue
            val_str = str(val).lower()
            # Check if any value appears in the question or hint
            if len(val_str) > 2 and val_str in " ".join(question_tokens | hint_tokens):
                score += 1.5

    # Sample row matches — check if question entities appear in sample data
    question_lower = " ".join(question_tokens)
    hint_lower = " ".join(hint_tokens)
    for row in table.get("sample_rows", []):
        for col_name, val in row.items():
            if val is None:
                continue
            val_str = str(val).lower()
            if len(val_str) > 3:
                if val_str in question_lower or val_str in hint_lower:
                    score += 1.0

    # Hint-specific boost — hints often name exact columns
    for col in table["columns"]:
        col_name_lower = col["name"].lower()
        # Check if column name appears as substring in hint
        for hint_tok in hint_tokens:
            if len(hint_tok) > 3 and hint_tok in col_name_lower:
                score += 2.0
            if len(col_name_lower) > 3 and col_name_lower in " ".join(hint_tokens):
                score += 2.0

    return score


def link_schema(
    profile: dict,
    question: str,
    hint: str = "",
    min_tables: int = 1,
    max_tables: int | None = None,
    score_threshold: float = 2.0,
) -> dict:
    """
    Filter a database profile to only include relevant tables.

    Args:
        profile: Database profile from profile_database()
        question: Natural language question
        hint: Evidence/hint string
        min_tables: Always include at least this many tables
        max_tables: Cap at this many tables (None = no cap)
        score_threshold: Minimum score to include a table

    Returns:
        Filtered profile dict with only relevant tables + their foreign keys
    """
    question_tokens = tokenize(question)
    hint_tokens = tokenize(hint) if hint else set()

    # Score all tables
    scored = []
    for table in profile["tables"]:
        s = score_table(table, question_tokens, hint_tokens)
        scored.append((s, table))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Select tables
    selected_names = set()
    selected_tables = []

    for s, table in scored:
        if len(selected_tables) < min_tables or s >= score_threshold:
            selected_tables.append(table)
            selected_names.add(table["name"])
        if max_tables and len(selected_tables) >= max_tables:
            break

    # Only add FK-connected tables if we have at least 2 selected tables
    # that need a bridge table, or if the selected table count is very small
    if len(selected_tables) <= 2:
        fk_additions = []
        for fk in profile.get("foreign_keys", []):
            if fk["from_table"] in selected_names and fk["to_table"] not in selected_names:
                for table in profile["tables"]:
                    if table["name"] == fk["to_table"]:
                        fk_additions.append(table)
                        selected_names.add(table["name"])
                        break
            elif fk["to_table"] in selected_names and fk["from_table"] not in selected_names:
                for table in profile["tables"]:
                    if table["name"] == fk["from_table"]:
                        fk_additions.append(table)
                        selected_names.add(table["name"])
                        break
        selected_tables.extend(fk_additions)

    # Filter foreign keys to only relevant ones
    relevant_fks = [
        fk for fk in profile.get("foreign_keys", [])
        if fk["from_table"] in selected_names and fk["to_table"] in selected_names
    ]

    return {
        "db_path": profile["db_path"],
        "db_name": profile["db_name"],
        "tables": selected_tables,
        "foreign_keys": relevant_fks,
    }


def format_linked_profile(profile: dict, question: str, hint: str = "") -> str:
    """Profile + link in one step, returns formatted string."""
    linked = link_schema(profile, question, hint)
    return format_profile(linked)


# ── CLI for testing ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test schema linking")
    parser.add_argument("--data_dir", type=str, default="data/dev")
    parser.add_argument("--task_idx", type=int, default=0)
    parser.add_argument("--show_all", action="store_true",
                        help="Show scores for all tables")

    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "dev.json")) as f:
        tasks = json.load(f)
    task = tasks[args.task_idx]

    db_id = task["db_id"]
    db_path = os.path.join(args.data_dir, "dev_databases", db_id, f"{db_id}.sqlite")
    profile = profile_database(db_path)

    question = task["question"]
    hint = task.get("evidence", "")
    question_tokens = tokenize(question)
    hint_tokens = tokenize(hint)

    print(f"Question: {question}")
    print(f"Hint: {hint}")
    print(f"DB: {db_id} ({len(profile['tables'])} tables)")
    print()

    if args.show_all:
        print("Table scores:")
        for table in profile["tables"]:
            s = score_table(table, question_tokens, hint_tokens)
            print(f"  {table['name']:30s} score={s:.1f}")
        print()

    linked = link_schema(profile, question, hint)
    print(f"Selected {len(linked['tables'])} tables: {[t['name'] for t in linked['tables']]}")
    print()

    formatted = format_profile(linked)
    print(f"Schema size: {len(formatted)} chars (was {len(format_profile(profile))} chars)")
    print()
    print(formatted[:500])
