"""
evaluate_gossip.py — Evaluation for Gossip SGD training.

Since there's no central parameter server, evaluation loads final weights
saved by each worker and evaluates them individually. Also averages all
workers' final weights for a "consensus" evaluation.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Reuse model from downpour
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "downpour"))
from model import get_model


def evaluate_test_set(model, data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss


def plot_training_curves(log_dir, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("worker_") and f.endswith("_metrics.json")]
    )
    if not metric_files:
        print("[EVALUATE] No metric files found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gossip SGD Training — 2-Worker Prototype", fontsize=14, fontweight="bold")

    colors = ["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"]

    for i, metric_file in enumerate(metric_files):
        filepath = os.path.join(log_dir, metric_file)
        with open(filepath) as f:
            metrics = json.load(f)

        epochs = [m["epoch"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        accuracies = [m["accuracy"] for m in metrics]
        worker_id = metric_file.split("_")[1]
        color = colors[i % len(colors)]

        ax1.plot(epochs, losses, marker="o", color=color, linewidth=2, label=f"Worker {worker_id}")
        ax2.plot(epochs, accuracies, marker="s", color=color, linewidth=2, label=f"Worker {worker_id}")

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Loss vs. Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Training Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs. Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[EVALUATE] Training curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gossip SGD Training")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-plot", type=str, default="./logs/training_curves.png")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gossip SGD — Final Evaluation")
    print("=" * 60)

    # Evaluate each worker's final model
    results = []
    all_state_dicts = []

    for wid in range(args.num_workers):
        weights_file = os.path.join(args.log_dir, f"worker_{wid}_final_weights.pt")
        if not os.path.exists(weights_file):
            print(f"[EVALUATE] Warning: {weights_file} not found, skipping worker {wid}")
            continue

        model = get_model()
        model.load_state_dict(torch.load(weights_file, weights_only=True))
        all_state_dicts.append(model.state_dict())

        accuracy, loss = evaluate_test_set(model, args.data_dir)
        results.append((wid, accuracy, loss))
        print(f"\n  Worker {wid}: Test Accuracy = {accuracy:.2f}%, Test Loss = {loss:.4f}")

    # Evaluate consensus model (average of all workers' final weights)
    if len(all_state_dicts) > 1:
        print(f"\n  --- Consensus Model (average of {len(all_state_dicts)} workers) ---")
        consensus_state = {}
        for name in all_state_dicts[0]:
            stacked = torch.stack([sd[name].float() for sd in all_state_dicts])
            consensus_state[name] = stacked.mean(dim=0)

        model = get_model()
        model.load_state_dict(consensus_state)
        consensus_acc, consensus_loss = evaluate_test_set(model, args.data_dir)
        print(f"  Consensus: Test Accuracy = {consensus_acc:.2f}%, Test Loss = {consensus_loss:.4f}")
    else:
        consensus_acc = results[0][1] if results else 0
        consensus_loss = results[0][2] if results else 0

    print(f"\n{'='*60}")
    print(f"  FINAL GOSSIP SGD RESULTS")
    for wid, acc, loss in results:
        print(f"  Worker {wid}: {acc:.2f}% accuracy, {loss:.4f} loss")
    if len(all_state_dicts) > 1:
        print(f"  Consensus:  {consensus_acc:.2f}% accuracy, {consensus_loss:.4f} loss")
    print(f"{'='*60}\n")

    # Plot training curves
    plot_training_curves(args.log_dir, args.output_plot)


if __name__ == "__main__":
    main()
