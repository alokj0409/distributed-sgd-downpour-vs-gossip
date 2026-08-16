"""
resilient_gossip_worker.py — Resilient Gossip SGD Worker with Dynamic Peer Pruning & Fault Simulation.

Extends standard Gossip worker with:
1. Dynamic Peer Pruning: Automatically detects failed/crashed peers on RPC errors and removes them from candidate pool.
2. Latency Injection: Simulates network delays using FaultInjector.
3. Crash Trigger: Simulates hardware failure by terminating process at specified epoch/batch.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import threading
from collections import OrderedDict
from concurrent import futures

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

import grpc

# Import protobuf & downpour modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "gossip", "generated"))
import gossip_service_pb2 as gs_pb2
import gossip_service_pb2_grpc as gs_grpc

sys.path.insert(0, os.path.join(PROJECT_ROOT, "downpour"))
from model import get_model, serialize_state_dict, deserialize_tensor

from fault_injector import FaultInjector, terminate_process_by_pid


def setup_logger(worker_id: int):
    logger = logging.getLogger(f"ResilientGossip-W{worker_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(f"[RESILIENT-GOSSIP-W{worker_id} %(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


class ResilientGossipServicer(gs_grpc.GossipPeerServicer):
    def __init__(self, worker_id: int, model: nn.Module, lock: threading.Lock, logger, fault_injector: FaultInjector):
        super().__init__()
        self.worker_id = worker_id
        self.model = model
        self.lock = lock
        self.logger = logger
        self.fault_injector = fault_injector
        self.ready = False
        self.bytes_sent = 0
        self.bytes_recv = 0

    def Ping(self, request, context):
        return gs_pb2.PingResponse(responder_id=self.worker_id, ready=self.ready)

    def ExchangeWeights(self, request, context):
        self.fault_injector.inject_latency()
        sender_id = request.sender_id
        self.bytes_recv += request.ByteSize()

        with self.lock:
            current_state = self.model.state_dict()
            serialized = serialize_state_dict(current_state)

            peer_weights = {}
            for td in request.weights:
                name = td.meta.name
                shape = list(td.meta.shape)
                tensor = deserialize_tensor(td.data, shape)
                peer_weights[name] = tensor

            averaged_state = OrderedDict()
            for name in current_state:
                if name in peer_weights:
                    averaged_state[name] = (current_state[name] + peer_weights[name]) / 2.0
                else:
                    averaged_state[name] = current_state[name]

            self.model.load_state_dict(averaged_state)

        response = gs_pb2.WeightExchangeResponse(responder_id=self.worker_id, success=True)
        for name, shape, data in serialized:
            td = gs_pb2.TensorData(meta=gs_pb2.TensorMeta(name=name, shape=shape), data=data)
            response.weights.append(td)

        self.bytes_sent += response.ByteSize()
        return response


def get_data_loader(worker_id, num_workers, batch_size, data_dir):
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
    return DataLoader(Subset(full_dataset, list(range(start_idx, end_idx))), batch_size=batch_size, shuffle=True, num_workers=0)


def resilient_gossip_exchange(
    model: nn.Module,
    peer_stubs: dict,
    peer_id: int,
    worker_id: int,
    lock: threading.Lock,
    logger,
    fault_injector: FaultInjector,
) -> tuple:
    """Initiates weight exchange; removes dead peer from active candidate pool on RPC failure."""
    fault_injector.inject_latency()
    if peer_id not in peer_stubs:
        return False, 0, 0, False

    stub = peer_stubs[peer_id]
    with lock:
        current_state = model.state_dict()
        serialized = serialize_state_dict(current_state)

    request = gs_pb2.WeightExchangeRequest(sender_id=worker_id)
    for name, shape, data in serialized:
        request.weights.append(gs_pb2.TensorData(meta=gs_pb2.TensorMeta(name=name, shape=shape), data=data))

    bytes_sent = request.ByteSize()
    try:
        response = stub.ExchangeWeights(request, timeout=5.0)
        bytes_recv = response.ByteSize()
    except grpc.RpcError as e:
        logger.warning(f"[FAULT DETECTED] Peer {peer_id} unreachable ({e.code()}). Pruning from active gossip list.")
        del peer_stubs[peer_id]  # Dynamic Peer Pruning!
        return False, bytes_sent, 0, True  # Peer died

    if not response.success:
        return False, bytes_sent, bytes_recv, False

    peer_weights = {}
    for td in response.weights:
        name = td.meta.name
        shape = list(td.meta.shape)
        peer_weights[name] = deserialize_tensor(td.data, shape)

    with lock:
        current_state = model.state_dict()
        averaged_state = OrderedDict()
        for name in current_state:
            if name in peer_weights:
                averaged_state[name] = (current_state[name] + peer_weights[name]) / 2.0
            else:
                averaged_state[name] = current_state[name]
        model.load_state_dict(averaged_state)

    return True, bytes_sent, bytes_recv, False


def train(
    worker_id: int,
    num_workers: int,
    base_port: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gossip_every: int,
    latency_ms: float,
    crash_epoch: int,
    data_dir: str,
    log_dir: str,
):
    logger = setup_logger(worker_id)
    fault_injector = FaultInjector(latency_ms=latency_ms)
    device = "cpu"
    lock = threading.Lock()

    model = get_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    my_port = base_port + worker_id
    servicer = ResilientGossipServicer(worker_id, model, lock, logger, fault_injector)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    gs_grpc.add_GossipPeerServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{my_port}")
    server.start()

    peer_stubs = {}
    peer_channels = {}
    grpc_options = [("grpc.max_send_message_length", 100 * 1024 * 1024), ("grpc.max_receive_message_length", 100 * 1024 * 1024)]
    for pid in range(num_workers):
        if pid != worker_id:
            peer_port = base_port + pid
            channel = grpc.insecure_channel(f"localhost:{peer_port}", options=grpc_options)
            peer_stubs[pid] = gs_grpc.GossipPeerStub(channel)
            peer_channels[pid] = channel

    servicer.ready = True
    time.sleep(2)

    data_loader = get_data_loader(worker_id, num_workers, batch_size, data_dir)
    metrics = []
    os.makedirs(log_dir, exist_ok=True)
    global_batch = 0
    gossip_count = 0
    training_start = time.time()
    client_bytes_sent = 0
    client_bytes_recv = 0

    for epoch in range(epochs):
        if crash_epoch > 0 and (epoch + 1) == crash_epoch and worker_id == 1:
            logger.error(f"[SIMULATED NODE CRASH] Worker {worker_id} dying at epoch {epoch+1}!")
            server.stop(grace=0)
            sys.exit(1)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with lock:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += targets.size(0)
            epoch_correct += predicted.eq(targets).sum().item()

            global_batch += 1

            if global_batch % gossip_every == 0 and peer_stubs:
                peer_id = random.choice(list(peer_stubs.keys()))
                success, b_sent, b_recv, peer_died = resilient_gossip_exchange(
                    model, peer_stubs, peer_id, worker_id, lock, logger, fault_injector
                )
                if success:
                    gossip_count += 1
                    client_bytes_sent += b_sent
                    client_bytes_recv += b_recv

        epoch_time = time.time() - epoch_start
        epoch_acc = 100.0 * epoch_correct / epoch_total
        avg_epoch_loss = epoch_loss / len(data_loader)
        total_elapsed = time.time() - training_start
        throughput = epoch_total / epoch_time if epoch_time > 0 else 0.0
        total_bytes = client_bytes_sent + servicer.bytes_sent + client_bytes_recv + servicer.bytes_recv

        metrics.append({
            "epoch": epoch + 1,
            "loss": avg_epoch_loss,
            "accuracy": epoch_acc,
            "epoch_time": epoch_time,
            "total_elapsed": total_elapsed,
            "gossip_count": gossip_count,
            "active_peers": len(peer_stubs),
            "total_bytes": total_bytes,
            "throughput_samples_per_sec": throughput,
        })

    metrics_file = os.path.join(log_dir, f"worker_{worker_id}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    weights_file = os.path.join(log_dir, f"worker_{worker_id}_final_weights.pt")
    torch.save(model.state_dict(), weights_file)

    for ch in peer_channels.values():
        ch.close()
    server.stop(grace=1)


def main():
    parser = argparse.ArgumentParser(description="Resilient Gossip SGD Worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=50060)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gossip-every", type=int, default=1)
    parser.add_argument("--latency-ms", type=float, default=0.0, help="Artificial link latency in ms")
    parser.add_argument("--crash-epoch", type=int, default=0, help="Epoch to trigger synthetic worker 1 crash (0=none)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    train(
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        base_port=args.base_port,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gossip_every=args.gossip_every,
        latency_ms=args.latency_ms,
        crash_epoch=args.crash_epoch,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
