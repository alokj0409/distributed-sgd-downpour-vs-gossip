"""
worker.py — Downpour SGD Worker.

Each worker:
1. Connects to the parameter server via gRPC
2. Loads its shard of the CIFAR-10 dataset
3. Training loop: pull weights → forward → backward → push gradients
4. Logs training metrics (loss, accuracy) to a JSON file for visualization
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

import grpc

# Import generated protobuf/gRPC modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))
import parameter_server_pb2 as ps_pb2
import parameter_server_pb2_grpc as ps_grpc

from model import (
    get_model,
    serialize_gradients,
    deserialize_tensor,
    serialize_tensor,
)

# Configure logging per worker
def setup_logger(worker_id: int):
    logger = logging.getLogger(f"Worker-{worker_id}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            f"[WORKER-{worker_id} %(asctime)s] %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)
    return logger


def get_data_loader(
    worker_id: int, num_workers: int, batch_size: int, data_dir: str
) -> DataLoader:
    """
    Create a DataLoader for this worker's shard of the CIFAR-10 training set.

    The training set is evenly split among workers by index range.
    Each worker gets len(dataset) // num_workers consecutive samples.
    """
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

    full_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Split dataset among workers
    total_size = len(full_dataset)
    shard_size = total_size // num_workers
    start_idx = worker_id * shard_size
    end_idx = start_idx + shard_size if worker_id < num_workers - 1 else total_size
    indices = list(range(start_idx, end_idx))
    subset = Subset(full_dataset, indices)

    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    return loader


def pull_weights(stub: ps_grpc.ParameterServerStub, worker_id: int, model: nn.Module) -> tuple:
    """
    Pull the latest global weights from the parameter server and load them into the model.
    Returns (global_step, bytes_sent, bytes_recv).
    """
    request = ps_pb2.PullRequest(worker_id=worker_id)
    bytes_sent = request.ByteSize()
    response = stub.PullWeights(request)
    bytes_recv = response.ByteSize()

    # Deserialize and load weights into model
    state_dict = {}
    for tensor_data in response.weights:
        name = tensor_data.meta.name
        shape = list(tensor_data.meta.shape)
        tensor = deserialize_tensor(tensor_data.data, shape)
        state_dict[name] = tensor

    model.load_state_dict(state_dict)
    return response.global_step, bytes_sent, bytes_recv


def push_gradients(
    stub: ps_grpc.ParameterServerStub, worker_id: int, model: nn.Module
) -> tuple:
    """
    Serialize the model's current gradients and push them to the parameter server.
    Returns (success, bytes_sent, bytes_recv).
    """
    grad_list = serialize_gradients(model)

    # Build gRPC request
    request = ps_pb2.GradientUpdate(worker_id=worker_id)
    for name, shape, data in grad_list:
        tensor_data = ps_pb2.TensorData(
            meta=ps_pb2.TensorMeta(name=name, shape=shape),
            data=data,
        )
        request.gradients.append(tensor_data)

    bytes_sent = request.ByteSize()
    response = stub.PushGradients(request)
    bytes_recv = response.ByteSize()
    return response.success, bytes_sent, bytes_recv


def train(
    worker_id: int,
    server_address: str,
    num_workers: int,
    epochs: int,
    batch_size: int,
    pull_every: int,
    data_dir: str,
    log_dir: str,
):
    """
    Main training loop for a Downpour SGD worker.

    Args:
        worker_id: Unique ID for this worker (0-indexed)
        server_address: gRPC address of the parameter server
        num_workers: Total number of workers (for data sharding)
        epochs: Number of training epochs
        batch_size: Mini-batch size
        pull_every: Pull weights from server every N batches
        data_dir: Directory for CIFAR-10 data
        log_dir: Directory for training logs
    """
    logger = setup_logger(worker_id)
    device = "cpu"  # CPU-only for the prototype

    # Connect to parameter server
    logger.info(f"Connecting to parameter server at {server_address}...")
    channel = grpc.insecure_channel(
        server_address,
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    stub = ps_grpc.ParameterServerStub(channel)

    # Wait for server to be ready
    for attempt in range(30):
        try:
            grpc.channel_ready_future(channel).result(timeout=2)
            logger.info("Connected to parameter server!")
            break
        except grpc.FutureTimeoutError:
            logger.info(f"Waiting for server... (attempt {attempt + 1})")
    else:
        logger.error("Failed to connect to parameter server after 30 attempts")
        return

    # Initialize model and load data
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    data_loader = get_data_loader(worker_id, num_workers, batch_size, data_dir)

    logger.info(
        f"Training config: epochs={epochs}, batch_size={batch_size}, "
        f"pull_every={pull_every}, data_shard_size={len(data_loader.dataset)}"
    )

    # Training metrics log
    metrics = []
    os.makedirs(log_dir, exist_ok=True)

    global_batch = 0
    cumulative_bytes_sent = 0
    cumulative_bytes_recv = 0
    training_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 1. Pull latest weights from parameter server
            if global_batch % pull_every == 0:
                try:
                    _, b_sent, b_recv = pull_weights(stub, worker_id, model)
                    cumulative_bytes_sent += b_sent
                    cumulative_bytes_recv += b_recv
                except grpc.RpcError as e:
                    logger.warning(f"Failed to pull weights: {e.code()}")

            # 2. Forward pass
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 3. Backward pass
            loss.backward()

            # 4. Push gradients to parameter server
            try:
                _, b_sent, b_recv = push_gradients(stub, worker_id, model)
                cumulative_bytes_sent += b_sent
                cumulative_bytes_recv += b_recv
            except grpc.RpcError as e:
                logger.warning(f"Failed to push gradients: {e.code()}")

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += targets.size(0)
            epoch_correct += predicted.eq(targets).sum().item()

            global_batch += 1

            # Log every 50 batches
            if (batch_idx + 1) % 50 == 0:
                batch_acc = 100.0 * epoch_correct / epoch_total
                avg_loss = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - training_start
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(data_loader)} | "
                    f"Loss: {avg_loss:.4f} | Acc: {batch_acc:.2f}% | "
                    f"Time: {elapsed:.1f}s"
                )

        # End of epoch summary
        epoch_time = time.time() - epoch_start
        epoch_acc = 100.0 * epoch_correct / epoch_total
        avg_epoch_loss = epoch_loss / len(data_loader)
        total_elapsed = time.time() - training_start
        throughput = epoch_total / epoch_time if epoch_time > 0 else 0.0

        logger.info(
            f"=== Epoch {epoch+1} complete | Loss: {avg_epoch_loss:.4f} | "
            f"Acc: {epoch_acc:.2f}% | Time: {epoch_time:.1f}s | Throughput: {throughput:.1f} samples/s ==="
        )

        metrics.append(
            {
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "accuracy": epoch_acc,
                "epoch_time": epoch_time,
                "total_elapsed": total_elapsed,
                "global_batch": global_batch,
                "bytes_sent": cumulative_bytes_sent,
                "bytes_recv": cumulative_bytes_recv,
                "total_bytes": cumulative_bytes_sent + cumulative_bytes_recv,
                "throughput_samples_per_sec": throughput,
            }
        )

    # Save metrics to file
    metrics_file = os.path.join(log_dir, f"worker_{worker_id}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training complete! Metrics saved to {metrics_file}")

    channel.close()


def main():
    parser = argparse.ArgumentParser(description="Downpour SGD Worker")
    parser.add_argument("--worker-id", type=int, required=True, help="Worker ID (0-indexed)")
    parser.add_argument("--server-address", type=str, default="localhost:50051", help="Parameter server address")
    parser.add_argument("--num-workers", type=int, default=2, help="Total number of workers")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--pull-every", type=int, default=1, help="Pull weights every N batches")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 data directory")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Training logs directory")
    args = parser.parse_args()

    train(
        worker_id=args.worker_id,
        server_address=args.server_address,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pull_every=args.pull_every,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
