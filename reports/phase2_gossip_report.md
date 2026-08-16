# Phase 2 Report: Gossip SGD (Decentralized Peer-to-Peer Architecture)

## 1. Executive Summary
Phase 2 implements **Gossip SGD**, a fully decentralized peer-to-peer (P2P) stochastic gradient descent framework. Unlike Downpour SGD, Gossip SGD eliminates the central Parameter Server entirely. Each worker node maintains its own local model state, performs local SGD optimization on its data shard, and periodically exchanges and averages model weights with randomly selected peer nodes.

---

## 2. System Architecture & Components

```
+---------------------------+    gRPC Peer Exchange    +---------------------------+
|      Worker 0 Node        | <======================> |      Worker 1 Node        |
|  - Dual Server + Client   |     W_new = (W_0+W_1)/2  |  - Dual Server + Client   |
|  - Shard 0 (CIFAR-10)     |                          |  - Shard 1 (CIFAR-10)     |
|  - Local SGD Optimizer    |                          |  - Local SGD Optimizer    |
+---------------------------+                          +---------------------------+
               ^                                                   ^
               |               gRPC Peer Exchange                  |
               +===================================================+
```

### Key Components:
- **`gossip/gossip_worker.py`**: Dual-role node operating as both a gRPC server (handling inbound weight exchange requests from peers) and a gRPC client (initiating outbound gossip exchanges).
- **`gossip/proto/gossip_service.proto`**: Interface definition for peer-to-peer weight exchange (`ExchangeWeights`, `Ping`).
- **`gossip/run_gossip.py`**: Local process launcher orchestrating $N$ peer workers with designated port ranges.
- **`gossip/evaluate_gossip.py`**: Evaluates individual worker models on the CIFAR-10 test set and computes a **Consensus Model** (averaging state dicts across all $N$ workers).

---

## 3. Communication Protocol (gRPC / Protobuf)

### Service Schema:
```protobuf
service GossipPeer {
    rpc Ping (PingRequest) returns (PingResponse);
    rpc ExchangeWeights (WeightExchangeRequest) returns (WeightExchangeResponse);
}
```

### Protocol Mechanics:
1. **Dual Servicer / Client Structure**: Each node listens on its assigned port (`base_port + worker_id`) while maintaining client stubs for all other peer nodes.
2. **Atomic Weight Averaging**:
   - Initiator node selects a random peer node.
   - Initiator sends its current state dict weights to peer.
   - Peer returns its state dict weights and updates its local weights to:
     $$\mathbf{W}_{\text{peer}}^{(new)} = \frac{\mathbf{W}_{\text{peer}} + \mathbf{W}_{\text{initiator}}}{2}$$
   - Initiator receives peer weights and updates its local weights to:
     $$\mathbf{W}_{\text{initiator}}^{(new)} = \frac{\mathbf{W}_{\text{initiator}} + \mathbf{W}_{\text{peer}}}{2}$$

---

## 4. Consensus Dynamics & Training Mechanics

- **Local SGD Step**: Each worker runs standard SGD with momentum on its local dataset shard:
  $$\mathbf{W}_i^{\left(t+\frac{1}{2}\right)} = \mathbf{W}_i^{(t)} - \eta \nabla L_i\left(\mathbf{W}_i^{(t)}\right)$$
- **Gossip Averaging Step**: Every `gossip_every` batches (default: 1), worker $i$ exchanges weights with randomly chosen peer $j$:
  $$\mathbf{W}_i^{(t+1)} = \frac{\mathbf{W}_i^{\left(t+\frac{1}{2}\right)} + \mathbf{W}_j^{\left(t+\frac{1}{2}\right)}}{2}$$
- **Consensus Aggregation**: Over training epochs, pairwise gossip exchanges cause local models across the cluster to converge toward a shared parameter manifold without requiring a central parameter server.

---

## 5. Deployment & Containerization

Containerized deployment is defined in `docker-compose.gossip.yml`:
- **Peer Services (`worker-0`, `worker-1`, ...)**: Peer-to-peer network mesh with exposed ports `50060`, `50061`, etc.
- **Dynamic Peer Discovery**: Automated via command line arguments (`--peers worker-0:50060,worker-1:50061`).
- **Compose Generator (`generate_compose.py`)**: CLI utility to generate compose files for arbitrary node topologies (`python generate_compose.py --mode gossip --num-workers N`).

---

## 6. Phase 2 Summary Checklist

| Requirement | Implementation Status | Notes |
| :--- | :--- | :--- |
| **Peer-to-Peer Node** | Completed (`gossip_worker.py`) | Dual gRPC server & client architecture |
| **Weight Averaging** | Completed (`gossip_worker.py`) | Pairwise atomic model parameter averaging |
| **Consensus Evaluation** | Completed (`evaluate_gossip.py`) | Aggregates $N$-worker state dicts into consensus model |
| **P2P Orchestration** | Completed (`run_gossip.py`) | Multi-process worker launcher with ready ping checking |
| **Docker Compose** | Completed (`docker-compose.gossip.yml`) | Decentralized bridge network topology |
