"""
resilient_downpour_worker.py — Resilient Downpour SGD Worker with Fault Handling & Latency Injection.

Extends standard Downpour SGD worker with:
1. gRPC Retry & Server Crash Detection: Retries failed pushes/pulls to Parameter Server.
2. Latency Injection: Simulates compute/network stragglers.
3. Crash Trigger: Simulates worker or PS failure.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "downpour", "generated"))
import parameter_server_pb2 as ps_pb2
import parameter_server_pb2_grpc as ps_grpc

sys.path.insert(0, os.path.join(PROJECT_ROOT, "downpour"))
from model import get_model, serialize_tensor, deserialize_tensor, serialize_gradients
from fault_injector import FaultInjector


def setup_logger(worker_id: int):
    logger = logging.getLogger(f"ResilientDownpour-W{worker_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(f"[RESILIENT-DOWNPOUR-W{worker_id} %(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def pull_weights_with_retry(stub, worker_id, model, logger, retries=3) -> bool:
    for attempt in range(retries):
        try:
            req = ps_pb2.PullRequest(worker_id=worker_id)
            resp = stub.PullWeights(req, timeout=5.0)
            server_weights = {}
            for td in resp.weights:
                tensor = deserialize_tensor(td.data, list(td.meta.shape))
                server_weights[td.meta.name] = tensor
            model.load_state_dict(server_weights)
            return True
        except grpc.RpcError as e:
            logger.warning(f"[PULL RETRY {attempt+1}/{retries}] Server error: {e.code()}")
            time.sleep(1.0)
    logger.error("[SYSTEM HALT] Parameter Server unreachable after retries!")
    return False


def push_gradients_with_retry(stub, worker_id, model, logger, retries=3) -> bool:
    update = ps_pb2.GradientUpdate(worker_id=worker_id)
    for name, shape, data in serialize_gradients(model):
        update.gradients.append(ps_pb2.TensorData(meta=ps_pb2.TensorMeta(name=name, shape=shape), data=data))
    for attempt in range(retries):
        try:
            resp = stub.PushGradients(update, timeout=5.0)
            return resp.success
        except grpc.RpcError as e:
            logger.warning(f"[PUSH RETRY {attempt+1}/{retries}] Server error: {e.code()}")
            time.sleep(1.0)
    logger.error("[SYSTEM HALT] Parameter Server unreachable after retries!")
    return False


def train(
    worker_id: int,
    server_address: str,
    num_workers: int,
    epochs: int,
    batch_size: int,
    pull_every: int,
    latency_ms: float,
    crash_epoch: int,
    data_dir: str,
    log_dir: str,
):
    logger = setup_logger(worker_id)
    fault_injector = FaultInjector(latency_ms=latency_ms)
    device = "cpu"

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()

    channel = grpc.insecure_channel(
        server_address,
        options=[("grpc.max_send_message_length", 100 * 1024 * 1024), ("grpc.max_receive_message_length", 100 * 1024 * 1024)],
    )
    stub = ps_grpc.ParameterServerStub(channel)

    if not pull_weights_with_retry(stub, worker_id, model, logger):
        sys.exit(1)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    full_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    total_size = len(full_dataset)
    shard_size = total_size // num_workers
    start_idx = worker_id * shard_size
    end_idx = start_idx + shard_size if worker_id < num_workers - 1 else total_size
    data_loader = DataLoader(Subset(full_dataset, list(range(start_idx, end_idx))), batch_size=batch_size, shuffle=True, num_workers=0)

    metrics = []
    os.makedirs(log_dir, exist_ok=True)
    global_batch = 0
    training_start = time.time()

    for epoch in range(epochs):
        if crash_epoch > 0 and (epoch + 1) == crash_epoch and worker_id == 1:
            logger.error(f"[SIMULATED NODE CRASH] Worker {worker_id} dying at epoch {epoch+1}!")
            sys.exit(1)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            fault_injector.inject_latency()
            inputs, targets = inputs.to(device), targets.to(device)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if not push_gradients_with_retry(stub, worker_id, model, logger):
                logger.error("[SYSTEM HALT] Aborting training due to server failure.")
                sys.exit(1)

            global_batch += 1
            if global_batch % pull_every == 0:
                if not pull_weights_with_retry(stub, worker_id, model, logger):
                    logger.error("[SYSTEM HALT] Aborting training due to server failure.")
                    sys.exit(1)

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += targets.size(0)
            epoch_correct += predicted.eq(targets).sum().item()

        epoch_time = time.time() - epoch_start
        epoch_acc = 100.0 * epoch_correct / epoch_total
        avg_epoch_loss = epoch_loss / len(data_loader)
        throughput = epoch_total / epoch_time if epoch_time > 0 else 0.0

        metrics.append({
            "epoch": epoch + 1,
            "loss": avg_epoch_loss,
            "accuracy": epoch_acc,
            "epoch_time": epoch_time,
            "total_elapsed": time.time() - training_start,
            "throughput_samples_per_sec": throughput,
        })

    metrics_file = os.path.join(log_dir, f"worker_{worker_id}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    channel.close()


def main():
    parser = argparse.ArgumentParser(description="Resilient Downpour Worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--server-address", type=str, default="localhost:50051")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pull-every", type=int, default=1)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--crash-epoch", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    train(
        worker_id=args.worker_id,
        server_address=args.server_address,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pull_every=args.pull_every,
        latency_ms=args.latency_ms,
        crash_epoch=args.crash_epoch,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
