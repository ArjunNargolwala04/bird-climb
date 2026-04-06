"""
Download BIRD training databases to a Modal Volume.

Fetches 69 SQLite databases from HuggingFace (premai-io/birdbench)
and uploads data/train/train.json so everything is accessible during training.

Usage:
    modal run rl/setup_train_data.py
"""

import modal
import os

app = modal.App("bird-climb-data-setup")

VOLUME_NAME = "bird-climb-train-data"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
)


@app.function(
    image=image,
    volumes={"/train_data": volume},
    timeout=3600,
)
def download_train_databases():
    """Download BIRD train databases from HuggingFace to Modal Volume."""
    from huggingface_hub import HfApi, hf_hub_download
    import sqlite3

    api = HfApi()
    repo_id = "premai-io/birdbench"
    repo_type = "dataset"

    db_dir = "/train_data/train_databases"
    os.makedirs(db_dir, exist_ok=True)

    # List all .sqlite files in the train/train_databases directory
    print("Listing files in premai-io/birdbench...")
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(repo_id, repo_type=repo_type)
    sqlite_files = [f for f in all_files if f.startswith("train/train_databases/") and f.endswith(".sqlite")]
    print(f"Found {len(sqlite_files)} .sqlite files")

    downloaded = 0
    skipped = 0

    for rfilename in sqlite_files:
        # rfilename looks like "train/train_databases/db_name/db_name.sqlite"
        # We want to store as /train_data/train_databases/db_name/db_name.sqlite
        relative = rfilename.removeprefix("train/train_databases/")
        dest_path = os.path.join(db_dir, relative)

        if os.path.exists(dest_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=rfilename,
                repo_type=repo_type,
                local_dir="/tmp/birdbench_download",
            )
            # Copy to volume
            import shutil
            shutil.copy2(local_path, dest_path)
            downloaded += 1

            if downloaded % 10 == 0:
                print(f"  Downloaded {downloaded} databases...")
        except Exception as e:
            print(f"  ERROR downloading {rfilename}: {e}")

    print(f"\nDownload complete: {downloaded} new, {skipped} already existed")

    # Commit volume changes
    volume.commit()

    # Verify: count databases and sanity check one
    db_names = [
        d for d in os.listdir(db_dir)
        if os.path.isdir(os.path.join(db_dir, d))
    ]
    print(f"\nTotal database directories: {len(db_names)}")

    # Sanity check: pick the first database and run a query
    if db_names:
        test_db = db_names[0]
        test_path = os.path.join(db_dir, test_db, f"{test_db}.sqlite")
        if os.path.exists(test_path):
            conn = sqlite3.connect(test_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            print(f"Sanity check - {test_db}: {len(tables)} tables: {tables[:5]}")
        else:
            print(f"WARNING: Expected {test_path} but not found")

    return {"downloaded": downloaded, "skipped": skipped, "total_dbs": len(db_names)}


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub"),
    volumes={"/train_data": volume},
    timeout=120,
)
def upload_train_json():
    """Download train.json from HuggingFace and save to volume."""
    import json

    # Download from HuggingFace instead of mounting local file
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(
        "birdsql/bird23-train-filtered",
        "data/train-00000-of-00001.jsonl",
        repo_type="dataset",
    )

    # Convert JSONL to JSON with the right format
    tasks = []
    with open(local_path) as f:
        for i, line in enumerate(f):
            task = json.loads(line)
            task["question_id"] = i
            tasks.append(task)

    dest = "/train_data/train.json"
    with open(dest, "w") as f:
        json.dump(tasks, f, indent=2)
    volume.commit()

    with open(dest) as f:
        tasks = json.load(f)
    print(f"Uploaded train.json: {len(tasks)} tasks")

    # Show database distribution
    from collections import Counter
    db_counts = Counter(t["db_id"] for t in tasks)
    print(f"Unique databases: {len(db_counts)}")
    print("Top 10 databases by task count:")
    for db, count in db_counts.most_common(10):
        print(f"  {db}: {count} tasks")

    return {"num_tasks": len(tasks), "num_databases": len(db_counts)}


@app.local_entrypoint()
def main():
    print("=== Step 1: Upload train.json ===")
    result = upload_train_json.remote()
    print(f"Result: {result}")

    print("\n=== Step 2: Download train databases ===")
    result = download_train_databases.remote()
    print(f"Result: {result}")

    print("\nDone! Volume 'bird-climb-train-data' is ready.")
