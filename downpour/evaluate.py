"""
evaluate.py — Evaluation and visualization for Downpour SGD training.

After training completes:
1. Pulls final weights from the parameter server
2. Evaluates on the CIFAR-10 test set
3. Plots training curves from worker logs
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

import grpc

# Import generated protobuf/gRPC modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))
import parameter_server_pb2 as ps_pb2
import parameter_server_pb2_grpc as ps_grpc

from model import get_model, deserialize_tensor


def evaluate_test_set(model: nn.Module, data_dir: str) -> tuple:
    """
    Evaluate model on the CIFAR-10 test set.
    Returns (test_accuracy, test_loss).
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

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


def pull_final_weights(server_address: str, model: nn.Module) -> int:
    """Pull final weights from the parameter server."""
    channel = grpc.insecure_channel(
        server_address,
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    stub = ps_grpc.ParameterServerStub(channel)

    request = ps_pb2.PullRequest(worker_id=-1)  # -1 = evaluator
    response = stub.PullWeights(request)

    state_dict = {}
    for tensor_data in response.weights:
        name = tensor_data.meta.name
        shape = list(tensor_data.meta.shape)
        tensor = deserialize_tensor(tensor_data.data, shape)
        state_dict[name] = tensor

    model.load_state_dict(state_dict)
    channel.close()
    return response.global_step


def plot_training_curves(log_dir: str, output_path: str):
    """
    Plot training curves from worker log files.
    Creates a combined plot showing loss and accuracy over epochs for all workers.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Find all worker metric files
    metric_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("worker_") and f.endswith("_metrics.json")]
    )

    if not metric_files:
        print("[EVALUATE] No metric files found in logs directory")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Downpour SGD Training — 2-Worker Prototype", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, metric_file in enumerate(metric_files):
        filepath = os.path.join(log_dir, metric_file)
        with open(filepath) as f:
            metrics = json.load(f)

        epochs = [m["epoch"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        accuracies = [m["accuracy"] for m in metrics]
        worker_id = metric_file.split("_")[1]

        color = colors[i % len(colors)]

        # Loss plot
        ax1.plot(epochs, losses, marker="o", color=color, linewidth=2, label=f"Worker {worker_id}")

        # Accuracy plot
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
    parser = argparse.ArgumentParser(description="Evaluate Downpour SGD Training")
    parser.add_argument("--server-address", type=str, default="localhost:50051")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--output-plot", type=str, default="./logs/training_curves.png")
    args = parser.parse_args()

    print("=" * 60)
    print("  Downpour SGD — Final Evaluation")
    print("=" * 60)

    # 1. Pull final weights from parameter server
    model = get_model()
    try:
        global_step = pull_final_weights(args.server_address, model)
        print(f"\n[EVALUATE] Pulled final weights (global step: {global_step})")
    except grpc.RpcError as e:
        print(f"[EVALUATE] Could not connect to server ({e.code()}), loading from latest checkpoint if available")
        return

    # 2. Evaluate on test set
    print("[EVALUATE] Evaluating on CIFAR-10 test set...")
    test_accuracy, test_loss = evaluate_test_set(model, args.data_dir)
    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Global Steps:  {global_step}")
    print(f"{'='*60}\n")

    # 3. Plot training curves
    plot_training_curves(args.log_dir, args.output_plot)


if __name__ == "__main__":
    main()
