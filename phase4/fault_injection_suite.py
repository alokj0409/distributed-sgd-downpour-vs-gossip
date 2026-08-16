"""
fault_injection_suite.py — Automated Fault Tolerance & Failure Recovery Test Suite.

Executes live fault-injection experiments for viva and empirical benchmarking:
1. Worker Crash Test: Kills worker 1 mid-training at epoch 2.
2. Parameter Server Crash Test: Kills the central Parameter Server in Downpour SGD.
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, "downpour", "venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_worker_crash_gossip(num_workers: int = 3, epochs: int = 3, log_dir: str = "./logs/fault_gossip"):
    print("\n" + "=" * 70)
    print(f"  [FAULT INJECTION TEST 1] Gossip SGD Worker Crash (Worker 1 killed at Epoch 2)")
    print("=" * 70)
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    base_port = 50070
    for wid in range(num_workers):
        crash_epoch = 2 if wid == 1 else 0
        cmd = [
            VENV_PYTHON,
            os.path.join(SCRIPT_DIR, "resilient_gossip_worker.py"),
            "--worker-id", str(wid),
            "--num-workers", str(num_workers),
            "--base-port", str(base_port),
            "--epochs", str(epochs),
            "--crash-epoch", str(crash_epoch),
            "--log-dir", log_dir,
            "--data-dir", os.path.join(PROJECT_ROOT, "downpour", "data"),
        ]
        p = subprocess.Popen(cmd)
        procs.append((wid, p))
        print(f"[LAUNCHER] Started Resilient Gossip Worker {wid} (crash_epoch={crash_epoch})...")

    # Monitor processes
    start = time.time()
    while any(p.poll() is None for wid, p in procs):
        time.sleep(1)
        if time.time() - start > 300:
            break

    print("[FAULT INJECTION COMPLETE] Gossip workers finished.")
    print("Results stored in:", log_dir)


def run_ps_crash_downpour(num_workers: int = 2, epochs: int = 3, log_dir: str = "./logs/fault_downpour"):
    print("\n" + "=" * 70)
    print(f"  [FAULT INJECTION TEST 2] Downpour SGD Parameter Server Crash Test")
    print("=" * 70)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Start Server
    server_cmd = [VENV_PYTHON, os.path.join(PROJECT_ROOT, "downpour", "server.py"), "--port", "50080"]
    server_proc = subprocess.Popen(server_cmd)
    print("[LAUNCHER] Started Parameter Server on port 50080 (PID:", server_proc.pid, ")")
    time.sleep(3)

    # 2. Start Workers
    worker_procs = []
    for wid in range(num_workers):
        w_cmd = [
            VENV_PYTHON,
            os.path.join(SCRIPT_DIR, "resilient_downpour_worker.py"),
            "--worker-id", str(wid),
            "--server-address", "localhost:50080",
            "--num-workers", str(num_workers),
            "--epochs", str(epochs),
            "--log-dir", log_dir,
            "--data-dir", os.path.join(PROJECT_ROOT, "downpour", "data"),
        ]
        p = subprocess.Popen(w_cmd)
        worker_procs.append((wid, p))

    # Wait 20 seconds into training then KILL Parameter Server
    print("[FAULT INJECTION] Waiting 20 seconds into training before killing Parameter Server...")
    time.sleep(20)
    print("[FAULT INJECTOR] >>> KILLING PARAMETER SERVER (PID:", server_proc.pid, ") <<<")
    server_proc.kill()

    # Check worker response
    time.sleep(5)
    for wid, p in worker_procs:
        ret = p.poll()
        print(f"[FAULT OBSERVER] Worker {wid} state after PS crash: returncode = {ret} (None=running, non-zero=halted)")

    print("[FAULT INJECTION COMPLETE] Downpour Parameter Server crash test complete.")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Fault Injection Suite")
    parser.add_argument("--test", type=str, default="all", choices=["all", "worker-crash", "ps-crash"])
    args = parser.parse_args()

    if args.test in ["all", "worker-crash"]:
        run_worker_crash_gossip()

    if args.test in ["all", "ps-crash"]:
        run_ps_crash_downpour()


if __name__ == "__main__":
    main()
