"""
run_gossip.py — Launcher script for the Gossip SGD prototype.

Starts N workers (each acting as both gRPC server and client) as subprocesses.
No central parameter server is needed — workers communicate peer-to-peer.
"""

import argparse
import os
import subprocess
import sys
import time
import threading


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(SCRIPT_DIR, "..", "downpour", "venv", "Scripts", "python.exe")

# Fallback to current python if venv doesn't exist
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable


def start_worker(
    worker_id, num_workers, base_port, epochs, batch_size, lr, gossip_every, data_dir, log_dir
):
    cmd = [
        VENV_PYTHON,
        os.path.join(SCRIPT_DIR, "gossip_worker.py"),
        "--worker-id", str(worker_id),
        "--num-workers", str(num_workers),
        "--base-port", str(base_port),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--gossip-every", str(gossip_every),
        "--data-dir", data_dir,
        "--log-dir", log_dir,
    ]
    print(f"[LAUNCHER] Starting gossip worker {worker_id} (port {base_port + worker_id})...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def stream_output(proc, label):
    for line in proc.stdout:
        print(f"{line}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Gossip SGD Launcher")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=50060)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gossip-every", type=int, default=1, help="Gossip every N batches")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    processes = []

    try:
        # Start all workers
        worker_procs = []
        for i in range(args.num_workers):
            proc = start_worker(
                worker_id=i,
                num_workers=args.num_workers,
                base_port=args.base_port,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                gossip_every=args.gossip_every,
                data_dir=args.data_dir,
                log_dir=args.log_dir,
            )
            worker_procs.append(proc)
            processes.append((f"worker-{i}", proc))

        print(f"[LAUNCHER] All {args.num_workers} gossip workers started (NO parameter server).")
        print(f"[LAUNCHER] Topology: peer-to-peer weight averaging every {args.gossip_every} batch(es)")
        print("=" * 70)

        # Stream output from all processes
        threads = []
        for label, proc in processes:
            t = threading.Thread(target=stream_output, args=(proc, label), daemon=True)
            t.start()
            threads.append(t)

        # Wait for all workers to finish
        for proc in worker_procs:
            proc.wait()

        print("=" * 70)
        print("[LAUNCHER] All gossip workers finished training!")

        # Run evaluation
        print("[LAUNCHER] Running final evaluation...")
        eval_cmd = [
            VENV_PYTHON,
            os.path.join(SCRIPT_DIR, "evaluate_gossip.py"),
            "--log-dir", args.log_dir,
            "--data-dir", args.data_dir,
            "--num-workers", str(args.num_workers),
        ]
        subprocess.run(eval_cmd, capture_output=False, text=True)

    except KeyboardInterrupt:
        print("\n[LAUNCHER] Interrupted! Shutting down...")

    finally:
        for label, proc in processes:
            if proc.poll() is None:
                print(f"[LAUNCHER] Stopping {label}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("[LAUNCHER] All processes stopped.")


if __name__ == "__main__":
    main()
