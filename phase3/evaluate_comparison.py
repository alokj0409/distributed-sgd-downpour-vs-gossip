"""
evaluate_comparison.py — Phase 3 Visualization Engine.

Loads training metrics from both Downpour SGD and Gossip SGD runs to generate
side-by-side comparative plots:
1. Loss & Accuracy vs. Epoch
2. Convergence (Loss & Accuracy) vs. Wall-Clock Time (seconds)
3. Cumulative Network Communication Overhead (MB) vs. Epoch
4. Worker Throughput (Samples / sec) vs. Epoch
"""

import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def load_worker_metrics(log_dir: str) -> dict:
    """Load all worker_X_metrics.json files from log_dir."""
    if not os.path.exists(log_dir):
        return {}
    
    files = sorted([f for f in os.listdir(log_dir) if f.startswith("worker_") and f.endswith("_metrics.json")])
    workers_metrics = {}
    for filename in files:
        worker_id = filename.split("_")[1]
        filepath = os.path.join(log_dir, filename)
        with open(filepath) as f:
            workers_metrics[worker_id] = json.load(f)
    return workers_metrics


def plot_accuracy_and_loss(dp_metrics: dict, gossip_metrics: dict, output_path: str):
    """Plot 1: Training Loss and Accuracy vs Epoch for Downpour vs Gossip."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Downpour SGD vs. Gossip SGD — Training Convergence", fontsize=15, fontweight="bold")

    dp_colors = ["#1E88E5", "#0D47A1", "#42A5F5"]
    gossip_colors = ["#E65100", "#FB8C00", "#FFB74D"]

    # Downpour plots
    for i, (wid, metrics) in enumerate(dp_metrics.items()):
        epochs = [m["epoch"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        accs = [m["accuracy"] for m in metrics]
        color = dp_colors[i % len(dp_colors)]
        ax1.plot(epochs, losses, marker="o", color=color, linewidth=2, label=f"Downpour Worker {wid}")
        ax2.plot(epochs, accs, marker="s", color=color, linewidth=2, label=f"Downpour Worker {wid}")

    # Gossip plots
    for i, (wid, metrics) in enumerate(gossip_metrics.items()):
        epochs = [m["epoch"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        accs = [m["accuracy"] for m in metrics]
        color = gossip_colors[i % len(gossip_colors)]
        ax1.plot(epochs, losses, marker="^", linestyle="--", color=color, linewidth=2, label=f"Gossip Worker {wid}")
        ax2.plot(epochs, accs, marker="d", linestyle="--", color=color, linewidth=2, label=f"Gossip Worker {wid}")

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Loss vs. Epoch", fontsize=13)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Training Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs. Epoch", fontsize=13)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATION] Saved convergence plot: {output_path}")


def plot_convergence_vs_time(dp_metrics: dict, gossip_metrics: dict, output_path: str):
    """Plot 2: Loss and Accuracy vs Wall-Clock Elapsed Time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Convergence Speed — Loss & Accuracy vs. Wall-Clock Time (s)", fontsize=15, fontweight="bold")

    dp_colors = ["#1E88E5", "#0D47A1"]
    gossip_colors = ["#E65100", "#FB8C00"]

    for i, (wid, metrics) in enumerate(dp_metrics.items()):
        times = [m["total_elapsed"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        accs = [m["accuracy"] for m in metrics]
        color = dp_colors[i % len(dp_colors)]
        ax1.plot(times, losses, marker="o", color=color, linewidth=2, label=f"Downpour Worker {wid}")
        ax2.plot(times, accs, marker="s", color=color, linewidth=2, label=f"Downpour Worker {wid}")

    for i, (wid, metrics) in enumerate(gossip_metrics.items()):
        times = [m["total_elapsed"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        accs = [m["accuracy"] for m in metrics]
        color = gossip_colors[i % len(gossip_colors)]
        ax1.plot(times, losses, marker="^", linestyle="--", color=color, linewidth=2, label=f"Gossip Worker {wid}")
        ax2.plot(times, accs, marker="d", linestyle="--", color=color, linewidth=2, label=f"Gossip Worker {wid}")

    ax1.set_xlabel("Elapsed Time (seconds)", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Loss vs. Wall-Clock Time", fontsize=13)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.set_xlabel("Elapsed Time (seconds)", fontsize=12)
    ax2.set_ylabel("Training Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs. Wall-Clock Time", fontsize=13)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATION] Saved convergence-vs-time plot: {output_path}")


def plot_communication_overhead(dp_metrics: dict, gossip_metrics: dict, output_path: str):
    """Plot 3: Cumulative Network Traffic Transferred (MB) vs Epoch."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle("Network Communication Overhead — Downpour vs. Gossip", fontsize=14, fontweight="bold")

    dp_colors = ["#1E88E5", "#0D47A1"]
    gossip_colors = ["#E65100", "#FB8C00"]

    for i, (wid, metrics) in enumerate(dp_metrics.items()):
        epochs = [m["epoch"] for m in metrics]
        bytes_total = [m.get("total_bytes", 0) / (1024 * 1024) for m in metrics]  # Convert to MB
        color = dp_colors[i % len(dp_colors)]
        ax.plot(epochs, bytes_total, marker="o", color=color, linewidth=2.5, label=f"Downpour Worker {wid}")

    for i, (wid, metrics) in enumerate(gossip_metrics.items()):
        epochs = [m["epoch"] for m in metrics]
        bytes_total = [m.get("total_bytes", 0) / (1024 * 1024) for m in metrics]
        color = gossip_colors[i % len(gossip_colors)]
        ax.plot(epochs, bytes_total, marker="^", linestyle="--", color=color, linewidth=2.5, label=f"Gossip Worker {wid}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cumulative Data Transferred (MB)", fontsize=12)
    ax.set_title("Cumulative Network Payload per Worker", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATION] Saved communication overhead plot: {output_path}")


def plot_throughput(dp_metrics: dict, gossip_metrics: dict, output_path: str):
    """Plot 4: Training Throughput (samples/sec) per Worker."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle("Worker Processing Throughput (Samples / sec)", fontsize=14, fontweight="bold")

    dp_colors = ["#1E88E5", "#0D47A1"]
    gossip_colors = ["#E65100", "#FB8C00"]

    for i, (wid, metrics) in enumerate(dp_metrics.items()):
        epochs = [m["epoch"] for m in metrics]
        tps = [m.get("throughput_samples_per_sec", 0) for m in metrics]
        color = dp_colors[i % len(dp_colors)]
        ax.plot(epochs, tps, marker="o", color=color, linewidth=2, label=f"Downpour Worker {wid}")

    for i, (wid, metrics) in enumerate(gossip_metrics.items()):
        epochs = [m["epoch"] for m in metrics]
        tps = [m.get("throughput_samples_per_sec", 0) for m in metrics]
        color = gossip_colors[i % len(gossip_colors)]
        ax.plot(epochs, tps, marker="^", linestyle="--", color=color, linewidth=2, label=f"Gossip Worker {wid}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Throughput (samples/sec)", fontsize=12)
    ax.set_title("Per-Worker Compute & Comm Efficiency", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATION] Saved throughput plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Comparative Plot Generator")
    parser.add_argument("--downpour-log-dir", type=str, required=True)
    parser.add_argument("--gossip-log-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dp_metrics = load_worker_metrics(args.downpour_log_dir)
    gossip_metrics = load_worker_metrics(args.gossip_log_dir)

    if not dp_metrics or not gossip_metrics:
        print("[EVALUATION ERROR] Metric logs missing for one or both runs!")
        return

    plot_accuracy_and_loss(dp_metrics, gossip_metrics, os.path.join(args.output_dir, "comparison_accuracy_loss.png"))
    plot_convergence_vs_time(dp_metrics, gossip_metrics, os.path.join(args.output_dir, "convergence_vs_time.png"))
    plot_communication_overhead(dp_metrics, gossip_metrics, os.path.join(args.output_dir, "communication_overhead.png"))
    plot_throughput(dp_metrics, gossip_metrics, os.path.join(args.output_dir, "throughput_comparison.png"))


if __name__ == "__main__":
    main()
