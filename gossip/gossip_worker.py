"""
gossip_worker.py — Gossip SGD Worker (Peer-to-Peer).

Each worker:
1. Runs a gRPC SERVER to handle incoming weight-exchange requests from peers
2. Acts as a gRPC CLIENT to initiate weight exchanges with randomly selected peers
3. Trains locally on its data shard
4. Periodically averages weights with a peer (gossip step)

Training loop:
    for each batch:
        1. Forward pass → compute loss → backward pass → local SGD step
        2. Every `gossip_every` batches:
           - Select a random peer
           - Send own weights, receive peer's weights
           - Average: W_new = (W_local + W_peer) / 2
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

# Import generated protobuf/gRPC modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))
import gossip_service_pb2 as gs_pb2
import gossip_service_pb2_grpc as gs_grpc

# Reuse model and serialization from downpour
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "downpour"))
from model import (
    get_model,
    serialize_state_dict,
    deserialize_state_dict,
    serialize_tensor,
    deserialize_tensor,
)


def setup_logger(worker_id: int):
    logger = logging.getLogger(f"Gossip-Worker-{worker_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                f"[GOSSIP-W{worker_id} %(asctime)s] %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# gRPC Servicer — handles incoming weight exchange requests from peers
# ---------------------------------------------------------------------------

class GossipPeerServicer(gs_grpc.GossipPeerServicer):
    """
    gRPC servicer that runs on each worker to handle incoming gossip requests.
    When a peer requests a weight exchange, this servicer:
    1. Returns the local model's current weights
    2. Averages the local weights with the received peer weights
    """

    def __init__(self, worker_id: int, model: nn.Module, lock: threading.Lock, logger):
        super().__init__()
        self.worker_id = worker_id
        self.model = model
        self.lock = lock
        self.logger = logger
        self.ready = False
        self.exchange_count = 0
        self.bytes_sent = 0
        self.bytes_recv = 0

    def Ping(self, request, context):
        return gs_pb2.PingResponse(responder_id=self.worker_id, ready=self.ready)

    def ExchangeWeights(self, request, context):
        """
        Handle incoming weight exchange request.
        1. Serialize current local weights to send back
        2. Average local weights with received peer weights
        """
        sender_id = request.sender_id
        self.bytes_recv += request.ByteSize()

        with self.lock:
            # Serialize current weights to return to the peer
            current_state = self.model.state_dict()
            serialized = serialize_state_dict(current_state)

            # Deserialize the peer's weights
            peer_weights = {}
            for td in request.weights:
                name = td.meta.name
                shape = list(td.meta.shape)
                tensor = deserialize_tensor(td.data, shape)
                peer_weights[name] = tensor

            # Average: W_new = (W_local + W_peer) / 2
            averaged_state = OrderedDict()
            for name in current_state:
                if name in peer_weights:
                    averaged_state[name] = (current_state[name] + peer_weights[name]) / 2.0
                else:
                    averaged_state[name] = current_state[name]

            self.model.load_state_dict(averaged_state)
            self.exchange_count += 1

        self.logger.debug(
            f"Exchanged weights with Worker {sender_id} "
            f"(exchange #{self.exchange_count})"
        )

        # Build response with our (pre-averaging) weights
        response = gs_pb2.WeightExchangeResponse(
            responder_id=self.worker_id, success=True
        )
        for name, shape, data in serialized:
            td = gs_pb2.TensorData(
                meta=gs_pb2.TensorMeta(name=name, shape=shape),
                data=data,
            )
            response.weights.append(td)

        self.bytes_sent += response.ByteSize()
        return response


# ---------------------------------------------------------------------------
# Data loading (same as downpour)
# ---------------------------------------------------------------------------

def get_data_loader(worker_id, num_workers, batch_size, data_dir):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])
    full_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    total_size = len(full_dataset)
    shard_size = total_size // num_workers
    start_idx = worker_id * shard_size
    end_idx = start_idx + shard_size if worker_id < num_workers - 1 else total_size
    indices = list(range(start_idx, end_idx))
    subset = Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------
# Gossip: initiate weight exchange with a peer
# ---------------------------------------------------------------------------

def gossip_exchange(
    model: nn.Module,
    peer_stub: gs_grpc.GossipPeerStub,
    worker_id: int,
    lock: threading.Lock,
    logger,
) -> tuple:
    """
    Initiate a weight exchange with a peer.
    Returns (success, bytes_sent, bytes_recv).
    """
    with lock:
        current_state = model.state_dict()
        serialized = serialize_state_dict(current_state)

    # Build request
    request = gs_pb2.WeightExchangeRequest(sender_id=worker_id)
    for name, shape, data in serialized:
        td = gs_pb2.TensorData(
            meta=gs_pb2.TensorMeta(name=name, shape=shape),
            data=data,
        )
        request.weights.append(td)

    bytes_sent = request.ByteSize()
    try:
        response = peer_stub.ExchangeWeights(request)
        bytes_recv = response.ByteSize()
    except grpc.RpcError as e:
        logger.warning(f"Gossip exchange failed: {e.code()}")
        return False, bytes_sent, 0

    if not response.success:
        return False, bytes_sent, bytes_recv

    # Deserialize peer's weights (these are the peer's weights BEFORE the peer averaged)
    peer_weights = {}
    for td in response.weights:
        name = td.meta.name
        shape = list(td.meta.shape)
        tensor = deserialize_tensor(td.data, shape)
        peer_weights[name] = tensor

    # Average: W_new = (W_local + W_peer) / 2
    with lock:
        current_state = model.state_dict()
        averaged_state = OrderedDict()
        for name in current_state:
            if name in peer_weights:
                averaged_state[name] = (current_state[name] + peer_weights[name]) / 2.0
            else:
                averaged_state[name] = current_state[name]
        model.load_state_dict(averaged_state)

    return True, bytes_sent, bytes_recv


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    worker_id: int,
    num_workers: int,
    base_port: int,
    peers_str: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gossip_every: int,
    data_dir: str,
    log_dir: str,
):
    logger = setup_logger(worker_id)
    device = "cpu"
    lock = threading.Lock()

    # Initialize model
    model = get_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Start gRPC server for this worker
    my_port = base_port + worker_id
    servicer = GossipPeerServicer(worker_id, model, lock, logger)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    gs_grpc.add_GossipPeerServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{my_port}")
    server.start()
    logger.info(f"Gossip peer server listening on port {my_port}")

    # Create gRPC stubs for all other peers
    peer_stubs = {}
    peer_channels = {}
    grpc_options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]
    if peers_str:
        peer_addresses = [p.strip() for p in peers_str.split(",") if p.strip()]
        for pid, peer_addr in enumerate(peer_addresses):
            # assign arbitrary peer ids for the connections
            actual_pid = pid if pid < worker_id else pid + 1
            channel = grpc.insecure_channel(peer_addr, options=grpc_options)
            peer_stubs[actual_pid] = gs_grpc.GossipPeerStub(channel)
            peer_channels[actual_pid] = channel
    else:
        for pid in range(num_workers):
            if pid != worker_id:
                peer_port = base_port + pid
                channel = grpc.insecure_channel(f"localhost:{peer_port}", options=grpc_options)
                peer_stubs[pid] = gs_grpc.GossipPeerStub(channel)
                peer_channels[pid] = channel

    # Wait for all peers to be ready
    servicer.ready = True
    logger.info(f"Waiting for {len(peer_stubs)} peers to be ready...")
    for pid, stub in peer_stubs.items():
        for attempt in range(60):
            try:
                resp = stub.Ping(gs_pb2.PingRequest(sender_id=worker_id))
                if resp.ready:
                    break
            except grpc.RpcError:
                pass
            time.sleep(0.5)
        else:
            logger.error(f"Peer {pid} not ready after 30 seconds!")
            return
    logger.info("All peers connected!")

    # Load data
    data_loader = get_data_loader(worker_id, num_workers, batch_size, data_dir)
    logger.info(
        f"Training config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, "
        f"gossip_every={gossip_every}, data_shard={len(data_loader.dataset)}"
    )

    # Training loop
    metrics = []
    os.makedirs(log_dir, exist_ok=True)
    global_batch = 0
    gossip_count = 0
    training_start = time.time()

    client_bytes_sent = 0
    client_bytes_recv = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 1. Local SGD step
            with lock:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += targets.size(0)
            epoch_correct += predicted.eq(targets).sum().item()

            global_batch += 1

            # 2. Gossip step: exchange weights with a random peer
            if global_batch % gossip_every == 0 and peer_stubs:
                peer_id = random.choice(list(peer_stubs.keys()))
                success, b_sent, b_recv = gossip_exchange(
                    model, peer_stubs[peer_id], worker_id, lock, logger
                )
                if success:
                    gossip_count += 1
                    client_bytes_sent += b_sent
                    client_bytes_recv += b_recv

            # Log every 50 batches
            if (batch_idx + 1) % 50 == 0:
                batch_acc = 100.0 * epoch_correct / epoch_total
                avg_loss = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - training_start
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(data_loader)} | "
                    f"Loss: {avg_loss:.4f} | Acc: {batch_acc:.2f}% | "
                    f"Gossips: {gossip_count} | Time: {elapsed:.1f}s"
                )

        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_acc = 100.0 * epoch_correct / epoch_total
        avg_epoch_loss = epoch_loss / len(data_loader)
        total_elapsed = time.time() - training_start
        throughput = epoch_total / epoch_time if epoch_time > 0 else 0.0

        total_bytes_sent = client_bytes_sent + servicer.bytes_sent
        total_bytes_recv = client_bytes_recv + servicer.bytes_recv

        logger.info(
            f"=== Epoch {epoch+1} complete | Loss: {avg_epoch_loss:.4f} | "
            f"Acc: {epoch_acc:.2f}% | Gossips: {gossip_count} | Time: {epoch_time:.1f}s | Throughput: {throughput:.1f} samples/s ==="
        )

        metrics.append({
            "epoch": epoch + 1,
            "loss": avg_epoch_loss,
            "accuracy": epoch_acc,
            "epoch_time": epoch_time,
            "total_elapsed": total_elapsed,
            "global_batch": global_batch,
            "gossip_count": gossip_count,
            "bytes_sent": total_bytes_sent,
            "bytes_recv": total_bytes_recv,
            "total_bytes": total_bytes_sent + total_bytes_recv,
            "throughput_samples_per_sec": throughput,
        })

    # Save metrics
    metrics_file = os.path.join(log_dir, f"worker_{worker_id}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training complete! Metrics saved to {metrics_file}")

    # Save final model weights
    weights_file = os.path.join(log_dir, f"worker_{worker_id}_final_weights.pt")
    torch.save(model.state_dict(), weights_file)
    logger.info(f"Final weights saved to {weights_file}")

    # Cleanup
    for ch in peer_channels.values():
        ch.close()
    server.stop(grace=2)


def main():
    parser = argparse.ArgumentParser(description="Gossip SGD Worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=50060, help="Base port; worker i uses base_port + i")
    parser.add_argument("--peers", type=str, default="", help="Comma-separated list of peer addresses (host:port)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gossip-every", type=int, default=1, help="Gossip every N batches")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    train(
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        base_port=args.base_port,
        peers_str=args.peers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gossip_every=args.gossip_every,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
