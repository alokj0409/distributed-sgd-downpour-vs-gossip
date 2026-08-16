# Phase 1 Report: Downpour SGD (Parameter Server Architecture)

## 1. Executive Summary
Phase 1 implements **Downpour SGD**, an asynchronous distributed stochastic gradient descent framework based on a centralized **Parameter Server (PS)** paradigm. The architecture separates model parameters from worker computation, allowing multiple worker nodes to independently compute gradients on local data shards and asynchronously update a centralized master model.

---

## 2. System Architecture & Components

```
+-------------------------------------------------------------+
|                   Parameter Server (PS)                     |
|  - Holds Master Weights (W)                                 |
|  - Maintains Adagrad Accumulators (G = G + g^2)            |
|  - Applies Async Updates: W = W - (lr / sqrt(G+eps)) * g    |
+-------------------------------------------------------------+
               ^                                 ^
   Push Grads /| Pull Weights        Push Grads /| Pull Weights
              v                                 v
+---------------------------+     +---------------------------+
|    Worker 0 Node          |     |    Worker 1 Node          |
|  - Shard 0 (CIFAR-10)     |     |  - Shard 1 (CIFAR-10)     |
|  - Forward / Backward     |     |  - Forward / Backward     |
+---------------------------+     +---------------------------+
```

### Key Components:
- **`downpour/server.py`**: Parameter Server hosting global model weights. Uses thread-safe Adagrad optimization to update master weights upon receiving gradient updates from workers.
- **`downpour/worker.py`**: Distributed worker node. Loads assigned dataset partition, pulls latest server weights, computes forward/backward pass, and pushes gradients to PS.
- **`downpour/model.py`**: 4-layer Convolutional Neural Network (CNN) designed for CIFAR-10 with custom tensor serialization/deserialization utilities over protobuf bytes.
- **`downpour/proto/parameter_server.proto`**: Interface definition for gRPC communication (`PushGradients`, `PullWeights`).
- **`downpour/run_downpour.py`**: Local process launcher orchestrating 1 Parameter Server and $N$ worker subprocesses.
- **`downpour/evaluate.py`**: Evaluates final master weights on the CIFAR-10 test set and plots worker training curves.

---

## 3. Communication Protocol (gRPC / Protobuf)

### Service Schema:
```protobuf
service ParameterServer {
    rpc PushGradients (GradientUpdate) returns (PushResponse);
    rpc PullWeights (PullRequest) returns (ModelWeights);
}
```

### Protocol Mechanics:
1. **PushGradients**: Worker serializes PyTorch gradient tensors into byte strings (`TensorData`) and streams them to the PS via `GradientUpdate`.
2. **PullWeights**: Worker requests master weights (`PullRequest`). The PS serializes master parameters into `ModelWeights` and returns them.

---

## 4. Optimization & Synchronization

- **Asynchronous Updates**: Workers operate independently without global barrier synchronization.
- **Adagrad Optimization**: Per-parameter adaptive learning rate implemented on the PS:
  $$\mathbf{G}^{(t)} = \mathbf{G}^{(t-1)} + \mathbf{g}^{(t)} \odot \mathbf{g}^{(t)}$$
  $$\mathbf{W}^{(t)} = \mathbf{W}^{(t-1)} - \frac{\eta}{\sqrt{\mathbf{G}^{(t)}} + \epsilon} \odot \mathbf{g}^{(t)}$$
- **Staleness**: In asynchronous Downpour SGD, gradients computed by worker $i$ at step $t$ may be applied to weights at step $t + \tau$. Adagrad dampens updates to frequently modified parameters, reducing gradient explosion under staleness.

---

## 5. Deployment & Containerization

Containerized deployment is defined in `docker-compose.downpour.yml`:
- **Parameter Server Service**: Listens on port `50051`.
- **Worker Services (`worker-0`, `worker-1`, ...)**: Automated scaling with shared volumes for CIFAR-10 dataset caching and metrics output.
- **Compose Generator (`generate_compose.py`)**: CLI utility to dynamically create `docker-compose.downpour.yml` for $N$ workers (`python generate_compose.py --mode downpour --num-workers N`).

---

## 6. Phase 1 Summary Checklist

| Requirement | Implementation Status | Notes |
| :--- | :--- | :--- |
| **Parameter Server** | Completed (`server.py`) | Thread-safe Adagrad parameter server |
| **Distributed Workers** | Completed (`worker.py`) | CIFAR-10 sharding & async gradient pushing |
| **Tensor Serialization** | Completed (`model.py`) | Raw byte float32 packing/unpacking |
| **Evaluation Suite** | Completed (`evaluate.py`) | Test accuracy evaluation & loss plotting |
| **Docker Compose** | Completed (`docker-compose.downpour.yml`) | Dynamic compose generator included |
