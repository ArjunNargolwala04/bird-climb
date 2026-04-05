"""
Database Profiler for BIRD Text-to-SQL

Extracts enriched schema information from SQLite databases:
- CREATE TABLE statements
- Foreign key relationships
- Sample rows per table
- Value distributions for low-cardinality columns

This is deterministic preprocessing that does NOT condition on the
question, hint, or answer — only on the database structure and contents.

Usage:
    from scaffold.profile import profile_database
    profile = profile_database("data/dev/dev_databases/california_schools/california_schools.sqlite")
"""

import sqlite3
import os
from typing import Optional


def profile_database(db_path: str, sample_rows: int = 3, max_distinct: int = 20) -> dict:
    """
    Profile a SQLite database and return enriched schema information.
    
    Args:
        db_path: Path to SQLite database file
        sample_rows: Number of sample rows per table
        max_distinct: Show distinct values for columns with fewer than this many unique values
    
    Returns:
        Dict with keys: tables, foreign_keys, db_path
        Each table has: name, create_sql, columns, sample_rows, column_values
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [row[0] for row in cursor.fetchall()]
    
    tables = []
    all_foreign_keys = []
    
    for table_name in table_names:
        table_info = profile_table(conn, table_name, sample_rows, max_distinct)
        tables.append(table_info)
        
        # Collect foreign keys
        for fk in table_info.get("foreign_keys", []):
            all_foreign_keys.append(fk)
    
    conn.close()
    
    return {
        "db_path": db_path,
        "db_name": os.path.basename(os.path.dirname(db_path)),
        "tables": tables,
        "foreign_keys": all_foreign_keys,
    }


def profile_table(conn: sqlite3.Connection, table_name: str, sample_rows: int = 3, max_distinct: int = 20) -> dict:
    """Profile a single table."""
    cursor = conn.cursor()
    
    # Get CREATE TABLE statement
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    row = cursor.fetchone()
    create_sql = row[0] if row else ""
    
    # Get column info
    cursor.execute(f"PRAGMA table_info(`{table_name}`)")
    columns = []
    for col in cursor.fetchall():
        columns.append({
            "name": col[1],     # column name
            "type": col[2],     # data type
            "notnull": bool(col[3]),
            "pk": bool(col[5]),
        })
    
    # Get foreign keys
    cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
    foreign_keys = []
    for fk in cursor.fetchall():
        foreign_keys.append({
            "from_table": table_name,
            "from_column": fk[3],       # from column
            "to_table": fk[2],          # referenced table
            "to_column": fk[4],         # referenced column
        })
    
    # Get sample rows
    try:
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT ?", (sample_rows,))
        col_names = [desc[0] for desc in cursor.description]
        samples = [dict(zip(col_names, row)) for row in cursor.fetchall()]
    except Exception:
        samples = []
        col_names = [c["name"] for c in columns]
    
    # Get row count
    try:
        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
        row_count = cursor.fetchone()[0]
    except Exception:
        row_count = 0
    
    # Get distinct values for low-cardinality columns
    column_values = {}
    for col in columns:
        col_name = col["name"]
        try:
            cursor.execute(f"SELECT COUNT(DISTINCT `{col_name}`) FROM `{table_name}`")
            n_distinct = cursor.fetchone()[0]
            
            if 0 < n_distinct <= max_distinct:
                cursor.execute(f"SELECT DISTINCT `{col_name}` FROM `{table_name}` ORDER BY `{col_name}` LIMIT ?", (max_distinct,))
                values = [row[0] for row in cursor.fetchall()]
                column_values[col_name] = values
        except Exception:
            continue
    
    return {
        "name": table_name,
        "create_sql": create_sql,
        "columns": columns,
        "foreign_keys": foreign_keys,
        "sample_rows": samples,
        "column_names": col_names,
        "row_count": row_count,
        "column_values": column_values,
    }


def format_profile(profile: dict) -> str:
    """
    Format a database profile into a string for inclusion in prompts.
    
    Produces a concise but information-rich representation:
    - CREATE TABLE statements
    - Foreign key relationships
    - Sample rows
    - Value distributions for low-cardinality columns
    """
    parts = []
    
    parts.append(f"Database: {profile['db_name']}")
    parts.append("")
    
    # Tables with CREATE statements and metadata
    for table in profile["tables"]:
        # CREATE TABLE
        parts.append(table["create_sql"] + ";")
        parts.append(f"-- {table['row_count']} rows")
        
        # Sample rows
        if table["sample_rows"]:
            parts.append(f"-- Sample rows from `{table['name']}`:")
            col_names = table["column_names"]
            for i, row in enumerate(table["sample_rows"]):
                vals = [str(row.get(c, "NULL"))[:50] for c in col_names]
                parts.append(f"--   {', '.join(vals)}")
        
        # Low-cardinality column values
        if table["column_values"]:
            for col_name, values in table["column_values"].items():
                str_values = [str(v) if v is not None else "NULL" for v in values]
                # Truncate if too long
                vals_str = ", ".join(str_values[:15])
                if len(str_values) > 15:
                    vals_str += ", ..."
                parts.append(f"-- `{col_name}` distinct values: [{vals_str}]")
        
        parts.append("")
    
    # Foreign key relationships
    if profile["foreign_keys"]:
        parts.append("-- Foreign Key Relationships:")
        for fk in profile["foreign_keys"]:
            parts.append(f"--   `{fk['from_table']}`.`{fk['from_column']}` -> `{fk['to_table']}`.`{fk['to_column']}`")
        parts.append("")
    
    return "\n".join(parts)


# ── CLI for testing ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile a BIRD database")
    parser.add_argument("db_path", type=str, help="Path to SQLite database file")
    parser.add_argument("--sample_rows", type=int, default=3)
    parser.add_argument("--max_distinct", type=int, default=20)
    
    args = parser.parse_args()
    
    profile = profile_database(args.db_path, args.sample_rows, args.max_distinct)
    formatted = format_profile(profile)
    print(formatted)