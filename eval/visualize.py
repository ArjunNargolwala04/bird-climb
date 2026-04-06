"""
Quick visualization of experiment results for presentation.

Usage:
    python -m eval.visualize
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_all_results():
    """Load key result files."""
    results = {}
    files = {
        "v1 baseline": "results/baseline_32b_detailed.json",
        "v2 prompt": "results/prompt_v2_detailed.json",
        "v2 + vote": "results/vote_v2_detailed.json",
        "v2 + GPT-4": "results/verify_v2_detailed.json",
        "7B baseline": "results/baseline_7b_detailed.json",
        "7B + GRPO": "results/rl_7b_detailed.json",
    }
    for name, path in files.items():
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    return results


def plot_accuracy_progression(results):
    """Bar chart of accuracy across experiments."""
    fig, ax = plt.subplots(figsize=(12, 5))

    names_32b = ["v1 baseline", "v2 prompt", "v2 + vote", "v2 + GPT-4"]
    names_7b = ["7B baseline", "7B + GRPO"]

    names = []
    accs = []
    colors = []

    for n in names_32b:
        if n in results:
            names.append(n)
            accs.append(results[n]["summary"]["execution_accuracy"] * 100)
            colors.append("#4C72B0")

    for n in names_7b:
        if n in results:
            names.append(n)
            accs.append(results[n]["summary"]["execution_accuracy"] * 100)
            colors.append("#DD8452")

    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Execution Accuracy (%)", fontsize=12)
    ax.set_title("BIRD Dev Set: Experiment Progression", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#4C72B0"), Patch(color="#DD8452")],
              ["32B model", "7B model"], loc="upper left")

    plt.tight_layout()
    plt.savefig("results/accuracy_progression.png", dpi=150)
    print("Saved results/accuracy_progression.png")


def plot_per_database(results):
    """Grouped bar chart of per-database accuracy for key runs."""
    fig, ax = plt.subplots(figsize=(14, 6))

    runs = ["v1 baseline", "v2 + vote"]
    run_colors = ["#A8D5E2", "#4C72B0"]

    available = [r for r in runs if r in results]
    if not available:
        return

    # Get database names from first available run
    dbs = sorted(results[available[0]]["summary"]["per_database_accuracy"].keys())

    x = np.arange(len(dbs))
    width = 0.35

    for i, run in enumerate(available):
        per_db = results[run]["summary"]["per_database_accuracy"]
        vals = [per_db.get(db, 0) * 100 for db in dbs]
        offset = (i - len(available) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=run, color=run_colors[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([db.replace("_", "\n") for db in dbs], fontsize=8)
    ax.set_ylabel("Execution Accuracy (%)", fontsize=12)
    ax.set_title("Per-Database Accuracy: Baseline vs Best", fontsize=14, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/per_database.png", dpi=150)
    print("Saved results/per_database.png")


def plot_error_breakdown(results):
    """Pie chart of error types from baseline."""
    if "v1 baseline" not in results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, run_name in [(ax1, "v1 baseline"), (ax2, "v2 + vote")]:
        if run_name not in results:
            continue

        res = results[run_name]["results"]
        from collections import Counter
        cats = Counter()
        for r in res:
            if r["match"]:
                cats["correct"] += 1
            elif r.get("pred_error"):
                cats["execution error"] += 1
            else:
                cats["wrong answer"] += 1

        labels = list(cats.keys())
        sizes = list(cats.values())
        colors_map = {"correct": "#55A868", "wrong answer": "#C44E52", "execution error": "#CCB974"}
        colors = [colors_map.get(l, "#999999") for l in labels]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.0f%%", colors=colors,
            textprops={"fontsize": 10}, startangle=90
        )
        ax.set_title(run_name, fontsize=12, fontweight="bold")

    fig.suptitle("Error Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/error_breakdown.png", dpi=150)
    print("Saved results/error_breakdown.png")


def plot_rl_training():
    """Plot RL training metrics from the log."""
    log_path = "results/train_v4_run.log"
    if not os.path.exists(log_path):
        return

    steps, losses, kls, rewards = [], [], [], []
    with open(log_path) as f:
        for line in f:
            if "Step " in line and "loss=" in line:
                parts = line.strip().split("|")
                try:
                    step = int(parts[0].split("Step")[1].split("/")[0].strip())
                    loss = float([p for p in parts if "loss=" in p][0].split("=")[1].strip())
                    kl = float([p for p in parts if "kl=" in p][0].split("=")[1].strip())
                    reward = float([p for p in parts if "reward=" in p][0].split("=")[1].strip())
                    steps.append(step)
                    losses.append(loss)
                    kls.append(kl)
                    rewards.append(reward)
                except (ValueError, IndexError):
                    continue

    if not steps:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, losses, color="#4C72B0", linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("GRPO Loss", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(steps, rewards, color="#55A868", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Training Reward", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("RL Training Metrics (7B, LR=2e-5, 2 epochs)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/rl_training.png", dpi=150)
    print("Saved results/rl_training.png")


if __name__ == "__main__":
    results = load_all_results()
    print(f"Loaded {len(results)} result files")

    plot_accuracy_progression(results)
    plot_per_database(results)
    plot_error_breakdown(results)
    plot_rl_training()

    print("\nAll visualizations saved to results/")
