# Centralized vs. Decentralized Distributed Training
### Downpour SGD (Parameter Server) vs. Gossip SGD (Peer-to-Peer) — A Head-to-Head Systems Comparison

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-orange.svg)](https://pytorch.org/)
[![gRPC 1.83](https://img.shields.io/badge/gRPC-1.83-green.svg)](https://grpc.io/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)

---

## 📌 Executive Overview

This repository contains a complete, from-scratch implementation and empirical benchmarking suite comparing two fundamental paradigms of distributed deep learning:
1. **Downpour SGD (Centralized)**: Parameter Server (PS) architecture with asynchronous worker parameter pulls and gradient pushes using Adagrad optimization.
2. **Gossip SGD (Decentralized)**: Peer-to-Peer (P2P) mesh topology with periodic pairwise model weight averaging across symmetric worker nodes.

Both paradigms train an identical 4-layer Convolutional Neural Network (586,250 parameters) on the **CIFAR-10** image classification dataset across 4 experimental phases:
- **Phase 1: Scalability Analysis** ($N=2, 4$ workers)
- **Phase 2: Live Fault Tolerance & Recovery** (Worker crash & Parameter Server process termination)
- **Phase 3: Network Communication Cost** (Payload volume tracking & throughput degradation)
- **Phase 4: Convergence Speed & Consensus Model Evaluation**

---

## 🚀 Key Experimental Findings

| Metric / Dimension | Downpour SGD (Parameter Server) | Gossip SGD (Peer-to-Peer Consensus) | Key Takeaway / Winner |
| :--- | :--- | :--- | :--- |
| **Final Training Loss** | `1.4328` | **`1.4259`** | Gossip SGD reaches slightly lower training loss |
| **Final Training Accuracy** | `47.65%` | **`48.17%`** | Gossip SGD worker accuracy (+0.52%) |
| **CIFAR-10 Test Accuracy** | `54.85%` (Master Model) | **`54.97%`** (Consensus Model) | Both paradigms converge to ~55% test accuracy |
| **Execution Speed ($N=2$)** | **`166.98 s`** | `258.01 s` | Downpour SGD is **1.54x faster** in wall-clock time |
| **Network Payload ($N=2$)** | **`10.49 GB`** | `20.99 GB` | Downpour uses 50% less payload (single push/pull) |
| **Scalability Trend ($N=2\to4$)**| **`122s → 103s` (-15.1%)** | `130s → 184s` (+41.6%) | Downpour benefits from smaller per-worker shards |
| **Fault Tolerance** | **Halts Irreversibly** | **Dynamic Peer Pruning** | Gossip continues training after worker failure; Downpour has SPOF |

---

## 📐 System Architecture

### 1. Centralized Downpour SGD
```
+-------------------------------------------------------------+
|                   Parameter Server (PS)                     |
|  - Holds Global Master Weights (W)                          |
|  - Maintains Adagrad Accumulator (G = G + g^2)              |
|  - W = W - (lr / sqrt(G+eps)) * g                           |
+-------------------------------------------------------------+
               ^                                 ^
   Push Grads /| Pull Weights        Push Grads /| Pull Weights
              v                                 v
+---------------------------+     +---------------------------+
|    Worker 0 Node          |     |    Worker 1 Node          |
|  - CIFAR-10 Shard 0       |     |  - CIFAR-10 Shard 1       |
|  - Forward / Backward     |     |  - Forward / Backward     |
+---------------------------+     +---------------------------+
```

### 2. Decentralized Gossip SGD
```
+---------------------------+    gRPC Peer Exchange    +---------------------------+
|      Worker 0 Node        | <======================> |      Worker 1 Node        |
|  - Dual Server + Client   |     W_new = (W_0+W_1)/2  |  - Dual Server + Client   |
|  - Local SGD Optimizer    |                          |  - Local SGD Optimizer    |
+---------------------------+                          +---------------------------+
```

---

## 📂 Repository Structure

```
├── downpour/                   # Phase 1: Downpour SGD Implementation
│   ├── server.py               # Adagrad Parameter Server (gRPC)
│   ├── worker.py               # Asynchronous Worker Process
│   ├── model.py                # PyTorch CNN & Tensor Serialization Helpers
│   ├── run_downpour.py         # Multi-process Local Launcher
│   └── proto/                  # Protobuf Service Definitions
├── gossip/                     # Phase 2: Gossip SGD Implementation
│   ├── gossip_worker.py        # Dual-role Peer Worker (Server + Client)
│   ├── evaluate_gossip.py      # Consensus Model Evaluator
│   ├── run_gossip.py           # Multi-peer Local Launcher
│   └── proto/                  # Protobuf Service Definitions
├── phase3/                     # Phase 3: Comparative Benchmark Suite
│   ├── run_comparison.py       # Orchestrator running both paradigms
│   ├── evaluate_comparison.py  # Plotting engine for convergence & throughput
│   └── report_generator.py     # Summary markdown report generator
├── phase4/                     # Phase 4: Fault Tolerance & Resilience Suite
│   ├── fault_injector.py       # Synthetic process termination & latency injector
│   ├── resilient_gossip_worker.py # Gossip worker with dynamic peer pruning
│   ├── resilient_downpour_worker.py # Downpour worker with PS failure retry
│   ├── fault_injection_suite.py # Automated live fault injection tests
│   ├── scalability_benchmark.py # Multi-worker scaling runner (N=2, 4)
│   └── evaluate_full_study.py  # Final plot generator
├── reports/                    # Documentation & LaTeX Paper
│   ├── phase1_downpour_report.md
│   ├── phase2_gossip_report.md
│   ├── phase3_comparative_report.md
│   ├── phase4_full_experiments_report.md
│   ├── paper.tex               # IEEE Conference Paper Source (LaTeX)
│   └── paper.md                # Markdown Version of Research Paper
├── Dockerfile                  # Container build specification
├── docker-compose.downpour.yml # Docker Compose setup for Downpour SGD
├── docker-compose.gossip.yml   # Docker Compose setup for Gossip SGD
├── generate_compose.py         # Dynamic Docker Compose configuration generator
└── README.md
```

---

## ⚡ Quick Start & Execution

### 1. Requirements
- Python 3.12+
- PyTorch 2.13+
- gRPC 1.83+

### 2. Local Multi-Process Execution

#### Run Comparative Benchmark Suite (Phase 3)
```bash
python phase3/run_comparison.py --num-workers 2 --epochs 3
```

#### Run Live Fault Injection Suite (Phase 4)
```bash
# Test 1: Worker crash mid-training (Gossip SGD dynamic pruning)
python phase4/fault_injection_suite.py --test worker-crash

# Test 2: Parameter Server crash mid-training (Downpour SGD SPOF halt)
python phase4/fault_injection_suite.py --test ps-crash
```

#### Run Multi-Worker Scalability Suite (Phase 4)
```bash
python phase4/scalability_benchmark.py --worker-counts 2,4 --epochs 2
```

### 3. Docker Compose Containerized Execution
```bash
# Generate Docker Compose configuration for 4 workers
python generate_compose.py --mode downpour --num-workers 4
docker-compose -f docker-compose.downpour.yml up --build
```

---

## 📄 IEEE Conference Paper

The full research paper documenting this project is included in [`reports/paper.md`](reports/paper.md) and formatted for IEEE conference submission in [`reports/paper.tex`](reports/paper.tex).

---

## 👨‍💻 Author

**Alok Jha**  
Department of Computer Science and Engineering  
Indian Institute of Information Technology, Pune, India  
Email: [alokj0409@gmail.com](mailto:alokj0409@gmail.com)  
GitHub: [@alokj0409](https://github.com/alokj0409)
