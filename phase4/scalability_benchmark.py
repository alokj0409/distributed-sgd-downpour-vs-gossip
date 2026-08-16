"""
scalability_benchmark.py — Multi-Worker Scalability Benchmark Suite (N = 2, 4, 8, 16).

Runs Downpour SGD and Gossip SGD across scaling worker counts to measure:
1. Time-to-Target Accuracy
2. Parameter Server Contention vs. P2P Scaling Efficiency
3. Network Traffic Volume (MB) per Worker Count
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, "downpour", "venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_scalability_experiment(worker_counts: list, epochs: int, base_log_dir: str):
    results = {"downpour": {}, "gossip": {}}
    os.makedirs(base_log_dir, exist_ok=True)

    print("=" * 70)
    print("  PHASE 4: MULTI-WORKER SCALABILITY BENCHMARK SUITE")
    print(f"  Worker Counts Evaluated: {worker_counts}")
    print(f"  Training Epochs: {epochs}")
    print("=" * 70)

    for n_workers in worker_counts:
        print(f"\n>>> BENCHMARKING WORKER COUNT: N = {n_workers} <<<")

        # 1. Downpour Run
        dp_log = os.path.join(base_log_dir, f"downpour_N{n_workers}")
        dp_cmd = [
            VENV_PYTHON,
            os.path.join(PROJECT_ROOT, "downpour", "run_downpour.py"),
            "--num-workers", str(n_workers),
            "--epochs", str(epochs),
            "--batch-size", str(64),
            "--log-dir", dp_log,
            "--data-dir", os.path.join(PROJECT_ROOT, "downpour", "data"),
        ]
        start_dp = time.time()
        res_dp = subprocess.run(dp_cmd)
        elapsed_dp = time.time() - start_dp
        print(f"[SCALABILITY] Downpour N={n_workers} finished in {elapsed_dp:.2f}s (returncode: {res_dp.returncode})")

        # 2. Gossip Run
        gossip_log = os.path.join(base_log_dir, f"gossip_N{n_workers}")
        gossip_cmd = [
            VENV_PYTHON,
            os.path.join(PROJECT_ROOT, "gossip", "run_gossip.py"),
            "--num-workers", str(n_workers),
            "--epochs", str(epochs),
            "--batch-size", str(64),
            "--base-port", str(50100 + n_workers * 20),
            "--log-dir", gossip_log,
            "--data-dir", os.path.join(PROJECT_ROOT, "downpour", "data"),
        ]
        start_gossip = time.time()
        res_gossip = subprocess.run(gossip_cmd)
        elapsed_gossip = time.time() - start_gossip
        print(f"[SCALABILITY] Gossip N={n_workers} finished in {elapsed_gossip:.2f}s (returncode: {res_gossip.returncode})")

        results["downpour"][str(n_workers)] = {"elapsed_sec": elapsed_dp, "status": res_dp.returncode == 0}
        results["gossip"][str(n_workers)] = {"elapsed_sec": elapsed_gossip, "status": res_gossip.returncode == 0}

    # Save scaling summary
    summary_path = os.path.join(base_log_dir, "scalability_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  [SCALABILITY SUITE COMPLETE] Results written to:", summary_path)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Scalability Benchmark")
    parser.add_argument("--worker-counts", type=str, default="2,4", help="Comma-separated worker counts (e.g., 2,4,8)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--log-dir", type=str, default=os.path.join(PROJECT_ROOT, "logs", "scalability"))
    args = parser.parse_args()

    worker_counts = [int(w.strip()) for w in args.worker_counts.split(",") if w.strip()]
    run_scalability_experiment(worker_counts, args.epochs, args.log_dir)


if __name__ == "__main__":
    main()
