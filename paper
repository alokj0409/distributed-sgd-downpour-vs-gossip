# Centralized vs. Decentralized Distributed Training: A Head-to-Head Empirical Comparison of Downpour SGD and Gossip SGD

**Author:** Alok Jha  
**Affiliation:** Department of Computer Science and Engineering — Indian Institute of Information Technology, Pune, India  
**Email:** alokj0409@gmail.com

---

## Abstract

Distributed Stochastic Gradient Descent (SGD) is foundational to large-scale deep learning. Two dominant paradigms exist: the *centralized Parameter Server* (PS) approach, exemplified by Downpour SGD, and *decentralized peer-to-peer* weight averaging, exemplified by Gossip SGD. Despite theoretical maturity, rigorous empirical comparison on identical hardware, datasets, and hyperparameters remains sparse. This paper presents a complete head-to-head empirical study implemented from scratch using PyTorch 2.13, gRPC 1.83, and Protocol Buffers in a Docker-based multi-process simulation on a single workstation-class host (Intel Core i5-12450H, 16 GB RAM, CPU-only). Both systems train a four-layer CNN (586,250 parameters) on CIFAR-10 across four experimental phases: convergence quality, worker scalability (N ∈ {2, 4}), network communication cost, and live fault-injection. Key findings: Downpour SGD completes three training epochs in 166.98 s versus 258.01 s for Gossip SGD (1.54× faster) and generates half the total network payload (11.0 GB vs. 22.0 GB). Both architectures converge to comparable CIFAR-10 test accuracy (~55% in 3 epochs). Gossip SGD tolerates worker crashes via autonomous peer pruning; Downpour SGD halts irreversibly on Parameter Server failure, confirming its single-point-of-failure (SPOF) characteristic.

**Keywords:** Distributed Machine Learning, Distributed SGD, Downpour SGD, Gossip SGD, Parameter Server, Fault Tolerance, Decentralized Optimization, gRPC, Peer-to-Peer Systems

---

## I. Introduction

The computational demands of modern deep-learning models consistently outpace single-machine capacity. Training neural networks requires distributing gradient computation across multiple workers [1]. This introduces a fundamental choice: *how* should model parameters be maintained, communicated, and updated?

### A. Centralized Parameter Server
The centralized paradigm, formalized in DistBelief and Downpour SGD [1], designates Parameter Server (PS) nodes as the authoritative keeper of global model weights. Workers asynchronously push gradients to the PS, which applies an Adagrad update [2] and serves the latest weights on demand. Simple and fast for early convergence, but concentrates network bandwidth and introduces a single point of failure.

### B. Decentralized Gossip-Based SGD
Decentralized approaches [3] eliminate the central server. In Gossip SGD, each worker trains on a local data shard and periodically exchanges model weights with a randomly selected peer. Both sides average their weights, driving the cluster toward consensus. Without a central bottleneck, this scales more gracefully and tolerates node failures without global disruption [6].

### C. Motivation and Contributions
Rigorous empirical comparison under controlled conditions is rarely reported. This paper addresses that gap with the following contributions:

1. Fully functional gRPC-based Downpour SGD implementation (PS + N async workers).
2. Fully functional gRPC-based Gossip SGD implementation (N symmetric peer workers, dual server/client roles).
3. Docker Compose simulation environment for reproducible multi-worker deployment.
4. Empirical measurements across four dimensions: convergence, scalability (N=2,4), network cost, and live fault injection.
5. Concrete, measurement-backed architecture selection recommendations.

---

## II. Background and Related Work

### A. Downpour SGD and DistBelief
Dean et al. [1] introduced DistBelief and Downpour SGD, where model replicas push gradients to sharded PS nodes applying Adagrad [2]. Li et al. [4] analyzed PS scalability, highlighting bandwidth concentration as a limiting factor at the server.

### B. Decentralized and Gossip-Based Optimization
Lian et al. [3] formally analyzed Decentralized Parallel SGD (D-PSGD), proving convergence for non-convex objectives under doubly-stochastic mixing matrices. Jin et al. [5] and Blot et al. [6] studied gossip-style weight averaging. Nedić and Ozdaglar [10] provided foundational theory for distributed subgradient convergence over gossip networks.

### C. Fault Tolerance
Synchronous all-reduce systems (e.g., Horovod [7]) are sensitive to crashes. Asynchronous PS systems handle worker failures but lose all state on PS failure [9]. Gossip systems have the theoretical advantage of no single critical node, empirically verified in this work.

---

## III. System Architecture and Design

### A. Model and Dataset
Both systems train identical four-layer CNN (`CifarCNN`) on CIFAR-10 [8]:

| Layer | Configuration |
| :--- | :--- |
| Conv Block 1 | Conv2d(3→32, 3×3, pad 1) → ReLU → MaxPool(2×2) |
| Conv Block 2 | Conv2d(32→64, 3×3, pad 1) → ReLU → MaxPool(2×2) |
| Conv Block 3 | Conv2d(64→64, 3×3, pad 1) → ReLU → MaxPool(2×2) |
| FC | Linear(1024→512) → ReLU → Dropout(0.25) → Linear(512→10) |

**Total parameters: 586,250.** The 50,000-sample CIFAR-10 training set is partitioned into N contiguous, disjoint shards.

### B. Communication Layer
All inter-process communication uses **gRPC 1.83** over **Protocol Buffers 3**. Raw float32 tensors are serialized via `numpy.tobytes()` and deserialized with `numpy.frombuffer`. Maximum gRPC message size: 100 MB.

### C. Downpour SGD Architecture
The Parameter Server (`server.py`) exposes two RPCs:
```protobuf
rpc PushGradients(GradientUpdate) returns (PushResponse);
rpc PullWeights(PullRequest) returns (ModelWeights);
```

Adagrad update applied on each received gradient **g** from worker *i*:

> **G** ← **G** + **g**²  
> **W** ← **W** − (η / √(**G** + ε)) ⊙ **g**

where η = 0.01, ε = 10⁻⁸. A `threading.Lock` ensures thread-safe concurrent access.

Each worker per mini-batch:
1. Pulls latest global weights from PS.
2. Forward pass → cross-entropy loss → backpropagation.
3. Pushes serialized gradients to PS.
4. Repeat (no global barrier synchronization).

### D. Gossip SGD Architecture
Each peer worker (`gossip_worker.py`) is simultaneously:
- A **gRPC server** listening on `base_port + worker_id`
- A **gRPC client** connected to all other peers

Service interface:
```protobuf
rpc ExchangeWeights(WeightExchangeRequest) returns (WeightExchangeResponse);
rpc Ping(PingRequest) returns (PingResponse);
```

Per mini-batch training loop for worker *i*:
1. Local SGD step: **W**ᵢ ← **W**ᵢ − η ∇L(**W**ᵢ), momentum = 0.9
2. Select random peer *j* from active candidate pool.
3. Send **W**ᵢ to peer *j*; receive peer's pre-averaging weights **W**ⱼ.
4. Both sides update: **W**ₖ ← (**W**ᵢ + **W**ⱼ) / 2, k ∈ {i, j}

### E. Consensus Model Evaluation
At the end of Gossip training:  
**W**_consensus = (1/N) Σ **W**ᵢ^(final)

Evaluated on CIFAR-10 test set to assess global model quality.

### F. Fault-Resilient Workers
- **Dynamic peer pruning:** On `grpc.RpcError (StatusCode.UNAVAILABLE)`, failed peer removed from active gossip pool.
- **PS retry with backoff:** Downpour workers retry failed push/pull RPCs up to 3× (1 s delay) before logging `[SYSTEM HALT]` and exiting.
- **Crash injection:** `--crash-epoch` flag causes designated worker to call `sys.exit(1)` at specified epoch boundary.

### G. Container Architecture
Both architectures ship Docker Compose files using `python:3.12-slim` base image with PyTorch (CPU), gRPC, and Matplotlib. Dynamic script `generate_compose.py` creates compose files for arbitrary N. Workers communicate over a Docker bridge network.

---

## IV. Experimental Methodology

### A. Hardware and Software Configuration

**TABLE I: Experimental Configuration**

| Parameter | Value |
| :--- | :--- |
| CPU | Intel Core i5-12450H (8C/12T, 12th Gen) |
| GPU | None (CPU-only) |
| RAM | 16 GB |
| OS | Windows 11 |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cpu |
| TorchVision | 0.28.0+cpu |
| gRPC | 1.83.0 |
| Protobuf | proto3 |
| Container | Docker Compose, bridge network |
| Dataset | CIFAR-10 (50K train / 10K test) |
| Model | CifarCNN (586,250 parameters) |
| Loss Function | Cross-Entropy |
| Mini-batch | 64 samples/worker |
| Learning Rate | 0.01 |
| PS Optimizer | Adagrad (ε = 1e-8) |
| Worker Optimizer | SGD (momentum = 0.9) |
| Gossip Frequency | Every 1 mini-batch |
| PS Pull Frequency | Every 1 mini-batch |
| Workers (Convergence) | N = 2 |
| Workers (Scalability) | N ∈ {2, 4} |
| Epochs (Convergence) | 3 |
| Epochs (Scalability) | 2 |

### B. Convergence Experiment
Both architectures trained for 3 epochs with N=2 workers. Each worker operated on a disjoint 25,000-sample CIFAR-10 shard. Training loss, accuracy, wall-clock time, bytes transferred, and sample throughput logged at each epoch boundary to JSON files. Final test accuracy evaluated using standard CIFAR-10 normalization.

### C. Scalability Experiment
Full benchmark pipeline executed for N ∈ {2, 4} with all other hyperparameters fixed. 2 epochs per run. Wall-clock time measured from process launch to completion of all workers.

### D. Fault-Tolerance Experiment
1. **Worker crash (Gossip SGD):** 3 Gossip workers launched; W₁ terminated via `sys.exit(1)` at epoch 2 boundary.
2. **PS crash (Downpour SGD):** PS allowed to process ~300 gradient steps, then OS process killed. Worker retry behavior recorded.

### E. Network Cost Measurement
Payload measured via `msg.ByteSize()` on all outbound and inbound gRPC protobuf messages at the worker level. All communication occurred over `localhost` (loopback).

---

## V. Results

### A. Convergence Results

**TABLE II: Per-Epoch Training Metrics (N=2, 3 Epochs)**

| System | Worker | Epoch | Loss | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| Downpour SGD | W0 | 1 | 1.8711 | 31.90 |
| | | 2 | 1.5141 | 44.32 |
| | | 3 | 1.4258 | 47.84 |
| | W1 | 1 | 1.8781 | 31.33 |
| | | 2 | 1.5311 | 43.71 |
| | | 3 | 1.4398 | 47.46 |
| Gossip SGD | W0 | 1 | 2.0747 | 22.53 |
| | | 2 | 1.6471 | 39.62 |
| | | 3 | 1.4207 | **48.17** |
| | W1 | 1 | 2.0865 | 22.15 |
| | | 2 | 1.6587 | 39.08 |
| | | 3 | 1.4311 | **48.17** |

Downpour SGD converges faster in Epoch 1 (31.9% vs. 22.5% accuracy). Gossip SGD closes the gap by Epoch 3 (48.17% vs. 47.65%). Final test-set accuracies: Gossip W0 = 55.29%, Downpour master = 54.85%, Gossip consensus = 54.97% — all within 0.44 percentage points.

📊 *Fig. 1* `comparison_accuracy_loss.png` — Training loss (left) and accuracy (right) vs. epoch. Downpour SGD (blue) converges faster in early epochs; both systems reach near-equivalent loss by epoch 3.

📊 *Fig. 2* `convergence_vs_time.png` — Training loss (left) and accuracy (right) vs. wall-clock time. Downpour completes in 166.98 s; Gossip requires 258.01 s (1.54× longer).

### B. Scalability Results

**TABLE III: Scalability — Wall-Clock Time vs. Worker Count (2 Epochs)**

| System | Workers (N) | Time (s) | Δ vs. N=2 |
| :--- | :--- | :--- | :--- |
| Downpour SGD | 2 | 122.36 | baseline |
| Downpour SGD | 4 | **103.90** | −15.1% |
| Gossip SGD | 2 | 130.29 | baseline |
| Gossip SGD | 4 | 184.53 | **+41.6%** |

Downpour SGD decreases wall-clock time by 15.1% from N=2 to N=4 (smaller shards, asynchronous PS integration). Gossip SGD increases by 41.6% (blocking bidirectional exchange per batch with more peers competing for availability).

📊 *Fig. 3* `scalability_throughput_vs_workers.png` — Wall-clock time vs. worker count. Downpour time decreases; Gossip time increases.

### C. Network Communication Results

**TABLE IV: Cumulative Network Payload (N=2, 3 Epochs)**

| System | Worker | Sent (GB) | Recv (GB) | Total (GB) |
| :--- | :--- | :--- | :--- | :--- |
| Downpour SGD | W0 | 2.751 | 2.751 | 5.502 |
| | W1 | 2.751 | 2.751 | 5.502 |
| **Downpour Total** | | | | **11.004** |
| Gossip SGD | W0 | 5.502 | 5.502 | 11.004 |
| | W1 | 5.502 | 5.502 | 11.004 |
| **Gossip Total** | | | | **22.008** |

Gossip SGD generates exactly 2× the network payload per worker. Each gossip exchange sends a full state dict in *both* directions (`WeightExchangeRequest` + `WeightExchangeResponse`), while Downpour performs one gradient push and one weight pull per batch (both O(params)).

📊 *Fig. 4* `communication_overhead.png` — Cumulative payload (MB) per worker vs. epoch. Gossip (orange) accumulates 2× Downpour (blue).

📊 *Fig. 5* `throughput_comparison.png` — Per-worker throughput (samples/sec) vs. epoch. Downpour: 384–502 samp/s; Gossip: 264–310 samp/s.

### D. Fault-Tolerance Results

#### Gossip SGD — Worker Crash
W₀ and W₂ immediately caught `grpc.RpcError (StatusCode.UNAVAILABLE)` when W₁ was killed at epoch 2, pruned it from active peer pools, and continued training to completion.

**TABLE V: Gossip SGD Surviving Worker Metrics After W₁ Crash at Epoch 2**

| Worker | Epoch | Loss | Accuracy (%) | Active Peers |
| :--- | :--- | :--- | :--- | :--- |
| W0 | 1 | 2.264 | 16.24 | 2 |
| | 2 | 1.961 | 27.43 | **1** |
| | 3 | 1.741 | 35.53 | 1 |
| W2 | 1 | 2.267 | 15.60 | 2 |
| | 2 | 1.943 | 28.52 | **1** |
| | 3 | 1.737 | 36.45 | 0 |

The Active Peers column confirms dynamic pruning: 2 active peers in epoch 1, drops to 1 after W₁'s crash in epoch 2. Training **never halted**.

📊 *Fig. 6* `fault_tolerance_recovery.png` — Surviving Gossip SGD worker accuracy after W₁ crash at epoch 2 (dashed red line). Both workers continue improving without interruption.

#### Downpour SGD — Parameter Server Crash
After ~300 gradient steps, the PS was killed. Workers W₀ and W₁ each retried 3× before halting:
```
[PUSH RETRY 1/3] StatusCode.UNAVAILABLE
[PUSH RETRY 2/3] StatusCode.UNAVAILABLE
[PUSH RETRY 3/3] StatusCode.UNAVAILABLE
[SYSTEM HALT] Parameter Server unreachable
```
Both workers exited with non-zero return codes. No model checkpoint saved. This empirically confirms the Parameter Server as a **single point of failure (SPOF)** for Downpour SGD.

---

## VI. Discussion

### A. Convergence Quality
Downpour SGD holds a significant early-epoch advantage (31.9% vs. 22.5% at epoch 1) because its Adagrad PS immediately aggregates gradient information from both workers per batch. Gossip SGD requires multiple exchange rounds to propagate information across the cluster. By epoch 3, both achieve comparable accuracy, suggesting the gap diminishes with longer training.

### B. Scalability: Diverging Trajectories
Downpour benefits from additional workers via smaller per-shard computation without increasing PS computation time. Gossip degrades because each mini-batch requires a blocking bidirectional gRPC round-trip, and with N=4, peer availability contention grows wall-clock time by 41.6%. Reducing `gossip_every` (e.g., to every 10 batches) would reduce this overhead at the cost of slower parameter mixing.

### C. Network Cost: Structural 2× Overhead
The 2× Gossip payload penalty is architectural, not incidental. At large N in a real multi-node cluster, Downpour's centralized server becomes a bandwidth bottleneck. For production billion-parameter models, this bottleneck would be severe; here it is not yet observable due to the small model size (2.34 MB/tensor).

### D. Fault Tolerance: Fundamental Architectural Difference
Gossip SGD's peer-pruning mechanism requires no external coordination and degrades gracefully. Downpour SGD has no recovery path when its PS is killed — a structural property of the centralized paradigm, mitigated in production through PS replication (not implemented here).

### E. Connecting the Four Phases
The four experiments tell a coherent systems story: Downpour SGD achieves faster convergence and lower network overhead for small N, but concentrates risk and will face bandwidth saturation at large N. Gossip SGD pays a constant 2× network overhead with worse scaling efficiency under per-batch gossip, but tolerates failures gracefully. Architecture selection depends on deployment priorities.

---

## VII. Limitations

1. **Single-machine simulation:** All experiments used process-level parallelism on one host; physical network latency and bandwidth caps were absent.
2. **CPU-only training:** Throughput numbers are dominated by CPU compute; GPU-accelerated hardware shifts communication as a larger fraction of step time.
3. **Limited worker counts:** Only N ∈ {2, 4} tested; scalability trends beyond N=4 are extrapolated.
4. **No PS replication:** Production deployments shard across multiple PS nodes for fault tolerance.
5. **Fixed gossip frequency:** `gossip_every=1` maximizes mixing but amplifies communication overhead; adaptive intervals not evaluated.
6. **No live network latency injection:** No live `tc netem` or equivalent applied to loopback interfaces.
7. **Small model/dataset:** CIFAR-10 with 586K parameters; behavior may differ for billion-parameter models.

---

## VIII. Conclusion

This paper presented a complete, reproducible empirical comparison of Downpour SGD (Parameter Server) and Gossip SGD (Peer-to-Peer) across four experimental phases. Both systems were implemented from scratch in Python/PyTorch with gRPC, deployed via Docker Compose, and benchmarked on identical CPU-only hardware.

**Principal findings:**
- Both architectures converge to similar final accuracy (~55% CIFAR-10 test) within 3 epochs.
- Downpour SGD trains **1.54× faster** with **half the network payload** under two-worker conditions.
- Downpour SGD shows **improved scaling efficiency** from N=2 to N=4 (−15.1% wall-clock time) while Gossip SGD degrades (+41.6%).
- Gossip SGD **withstands worker crashes** without interruption via autonomous peer pruning.
- Downpour SGD **halts irreversibly on PS failure**, confirming the SPOF characteristic of centralized training.

**Recommendation:** For small, reliable clusters requiring fast convergence → Downpour SGD. For large-scale, failure-prone deployments where resilience matters → Gossip SGD.

---

## IX. Future Work

- Multi-machine deployment (Kubernetes / bare-metal) for scalability beyond N=4.
- GPU-accelerated workers to shift compute/communication ratio.
- Adaptive gossip frequency (e.g., every 10 batches) to investigate accuracy/overhead trade-offs.
- PS replication (sharded/replicated PS nodes) for fault-tolerant Downpour SGD.
- Byzantine fault tolerance under adversarial gradient conditions.
- Larger models (ResNet-50, GPT-style) and datasets (ImageNet).
- Structured gossip topologies (ring, hypercube) vs. random peer selection.
- Live network simulation via Linux `tc netem` for realistic WAN latency.

---

## References

[1] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, Q. Le, M. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, and A. Ng, "Large Scale Distributed Deep Networks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 25, 2012, pp. 1223–1231.

[2] J. Duchi, E. Hazan, and Y. Singer, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization," *J. Mach. Learn. Res.*, vol. 12, pp. 2121–2159, 2011.

[3] X. Lian, C. Zhang, H. Zhang, C.-J. Hsieh, W. Zhang, and J. Liu, "Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5330–5340.

[4] M. Li, D. G. Andersen, J. W. Park, A. J. Smola, A. Ahmed, V. Josifovski, J. Long, E. J. Shekita, and B.-Y. Su, "Scaling Distributed Machine Learning with the Parameter Server," in *Proc. USENIX OSDI*, 2014, pp. 583–598.

[5] P.-H. Jin, Q. Yuan, F. Iandola, and K. Keutzer, "How to Scale Distributed Deep Learning?" *arXiv preprint arXiv:1611.04581*, 2016.

[6] M. Blot, D. Picard, L. Chen, N. Thome, and M. Cord, "Gossip Training for Deep Learning," *arXiv preprint arXiv:1611.09726*, 2016.

[7] A. Sergeev and M. Del Balso, "Horovod: Fast and Easy Distributed Deep Learning in TensorFlow," *arXiv preprint arXiv:1802.05799*, 2018.

[8] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. Rep., Univ. of Toronto, 2009.

[9] Q. Ho, J. Cipar, H. Cui, S. Lee, J. K. Kim, P. B. Gibbons, G. A. Gibson, G. Ganger, and E. P. Xing, "More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 26, 2013.

[10] A. Nedić and A. Ozdaglar, "Distributed Subgradient Methods for Multi-Agent Optimization," *IEEE Trans. Autom. Control*, vol. 54, no. 1, pp. 48–61, Jan. 2009.
