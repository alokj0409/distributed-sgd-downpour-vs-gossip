"""
run_comparison.py — Phase 3 Benchmark Suite Orchestrator.

Runs both Downpour SGD and Gossip SGD training experiments sequentially under
identical conditions (same epochs, worker count, batch size, and dataset splits),
then triggers comparative evaluation and summary report generation.
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Use venv python if available
VENV_PYTHON = os.path.join(PROJECT_ROOT, "downpour", "venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable


def run_downpour_experiment(num_workers: int, epochs: int, batch_size: int, data_dir: str, log_dir: str):
    print("\n" + "=" * 70)
    print(f"  [PHASE 3 BENCHMARK] STEP 1/2: Downpour SGD ({num_workers} Workers, {epochs} Epochs)")
    print("=" * 70)
    downpour_script = os.path.join(PROJECT_ROOT, "downpour", "run_downpour.py")
    cmd = [
        VENV_PYTHON,
        downpour_script,
        "--num-workers", str(num_workers),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--data-dir", data_dir,
        "--log-dir", log_dir,
    ]
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    print(f"[BENCHMARK] Downpour SGD run completed in {elapsed:.2f} seconds with returncode {result.returncode}")
    return result.returncode == 0, elapsed


def run_gossip_experiment(num_workers: int, epochs: int, batch_size: int, data_dir: str, log_dir: str):
    print("\n" + "=" * 70)
    print(f"  [PHASE 3 BENCHMARK] STEP 2/2: Gossip SGD ({num_workers} Workers, {epochs} Epochs)")
    print("=" * 70)
    gossip_script = os.path.join(PROJECT_ROOT, "gossip", "run_gossip.py")
    cmd = [
        VENV_PYTHON,
        gossip_script,
        "--num-workers", str(num_workers),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--data-dir", data_dir,
        "--log-dir", log_dir,
    ]
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    print(f"[BENCHMARK] Gossip SGD run completed in {elapsed:.2f} seconds with returncode {result.returncode}")
    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Comparative Benchmark Suite")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of worker nodes")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size per worker")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "downpour", "data"), help="Dataset directory")
    parser.add_argument("--log-dir", type=str, default=os.path.join(PROJECT_ROOT, "logs"), help="Base directory for benchmark logs")
    args = parser.parse_args()

    downpour_log_dir = os.path.join(args.log_dir, "downpour")
    gossip_log_dir = os.path.join(args.log_dir, "gossip")

    os.makedirs(downpour_log_dir, exist_ok=True)
    os.makedirs(gossip_log_dir, exist_ok=True)

    print("=" * 70)
    print("  PHASE 3: DISTRIBUTED SGD COMPARATIVE BENCHMARK SUITE")
    print(f"  Architectures: Downpour SGD (Parameter Server) vs Gossip SGD (Peer-to-Peer)")
    print(f"  Workers: {args.num_workers} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 70)

    # 1. Run Downpour SGD
    success_dp, time_dp = run_downpour_experiment(
        args.num_workers, args.epochs, args.batch_size, args.data_dir, downpour_log_dir
    )

    # 2. Run Gossip SGD
    success_gossip, time_gossip = run_gossip_experiment(
        args.num_workers, args.epochs, args.batch_size, args.data_dir, gossip_log_dir
    )

    # 3. Generate Comparative Evaluation Plots & Report
    if success_dp and success_gossip:
        print("\n" + "=" * 70)
        print("  [PHASE 3 BENCHMARK] Generating Comparative Visualizations & Summary Report")
        print("=" * 70)
        
        eval_script = os.path.join(SCRIPT_DIR, "evaluate_comparison.py")
        subprocess.run([
            VENV_PYTHON, eval_script,
            "--downpour-log-dir", downpour_log_dir,
            "--gossip-log-dir", gossip_log_dir,
            "--output-dir", args.log_dir,
        ])

        report_script = os.path.join(SCRIPT_DIR, "report_generator.py")
        subprocess.run([
            VENV_PYTHON, report_script,
            "--downpour-log-dir", downpour_log_dir,
            "--gossip-log-dir", gossip_log_dir,
            "--data-dir", args.data_dir,
            "--num-workers", str(args.num_workers),
            "--output-dir", args.log_dir,
        ])

        print("\n" + "=" * 70)
        print("  [PHASE 3 BENCHMARK COMPLETE] All comparative metrics, plots & reports generated.")
        print(f"  Artifacts stored in: {args.log_dir}")
        print("=" * 70)
    else:
        print("\n[BENCHMARK ERROR] One or both benchmark runs failed. Please check logs.")


if __name__ == "__main__":
    main()
