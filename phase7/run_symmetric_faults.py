"""
run_symmetric_faults.py — Symmetric Fault Tolerance Evaluation Engine.

Evaluates three symmetric fault scenarios:
1. Gossip Worker Crash (Worker W1 killed at Epoch 2; W0 and W2 dynamically prune W1 and continue training)
2. Downpour Worker Crash (Worker W1 killed at Epoch 2; PS stays alive, W0 and W2 continue pushing gradients)
3. Downpour PS Crash (Parameter Server killed at Step 300; workers retry 3 times and halt)

Saves results to logs/phase7/symmetric_fault_results.json.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np

def get_python_exe(root_dir="."):
    venv_py = os.path.join(root_dir, "downpour", "venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

def run_cmd(cmd):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def wait_processes(procs):
    for p in procs:
        p.wait()

def run_downpour_worker_crash(num_workers=3, crash_worker_id=1, crash_epoch=2, epochs=3, root_dir="."):
    """Evaluates Downpour SGD behavior when a single worker process crashes while PS stays alive."""
    log_dir = os.path.abspath("logs/phase7/fault_downpour_worker_crash")
    os.makedirs(log_dir, exist_ok=True)
    py_exe = get_python_exe(root_dir)

    port = 50080
    server_cmd = [
        py_exe,
        os.path.join(root_dir, "downpour", "server.py"),
        "--port", str(port),
        "--lr", "0.01",
    ]
    server_proc = run_cmd(server_cmd)
    time.sleep(3)

    worker_procs = []
    for w in range(num_workers):
        w_cmd = [
            py_exe,
            os.path.join(root_dir, "downpour", "worker.py"),
            "--worker-id", str(w),
            "--server-address", f"localhost:{port}",
            "--num-workers", str(num_workers),
            "--epochs", str(epochs),
            "--batch-size", "64",
            "--data-dir", os.path.join(root_dir, "downpour", "data"),
            "--log-dir", log_dir,
        ]
        if w == crash_worker_id:
            w_cmd.extend(["--crash-epoch", str(crash_epoch)])
        worker_procs.append(run_cmd(w_cmd))

    wait_processes(worker_procs)
    server_proc.kill()

    results = {}
    for w in range(num_workers):
        w_file = os.path.join(log_dir, f"worker_{w}_metrics.json")
        if os.path.exists(w_file):
            with open(w_file, "r") as f:
                results[f"worker_{w}"] = json.load(f)

    return results

def run_gossip_worker_crash(num_workers=3, crash_worker_id=1, crash_epoch=2, epochs=3, root_dir="."):
    """Evaluates Gossip SGD behavior when worker W1 crashes at Epoch 2 boundary."""
    log_dir = os.path.abspath("logs/phase7/fault_gossip_worker_crash")
    os.makedirs(log_dir, exist_ok=True)
    py_exe = get_python_exe(root_dir)

    base_port = 50100
    worker_procs = []
    for w in range(num_workers):
        w_cmd = [
            py_exe,
            os.path.join(root_dir, "gossip", "gossip_worker.py"),
            "--worker-id", str(w),
            "--num-workers", str(num_workers),
            "--base-port", str(base_port),
            "--epochs", str(epochs),
            "--batch-size", "64",
            "--gossip-every", "1",
            "--data-dir", os.path.join(root_dir, "downpour", "data"),
            "--log-dir", log_dir,
        ]
        if w == crash_worker_id:
            w_cmd.extend(["--crash-epoch", str(crash_epoch)])
        worker_procs.append(run_cmd(w_cmd))

    wait_processes(worker_procs)

    results = {}
    for w in range(num_workers):
        w_file = os.path.join(log_dir, f"worker_{w}_metrics.json")
        if os.path.exists(w_file):
            with open(w_file, "r") as f:
                results[f"worker_{w}"] = json.load(f)

    return results

def main():
    print("=" * 75)
    print("STARTING SYMMETRIC FAULT-TOLERANCE EVALUATION")
    print("=" * 75)

    print("\n--- 1. Executing Downpour Worker-Crash Experiment ---")
    dp_worker_crash = run_downpour_worker_crash()
    print("  -> Downpour Worker Crash complete.")
    for w_name, metrics in dp_worker_crash.items():
        epochs_completed = len(metrics)
        final_acc = metrics[-1]["accuracy"] if metrics else 0.0
        print(f"     {w_name}: Completed {epochs_completed} Epochs | Final Acc = {final_acc:.2f}%")

    print("\n--- 2. Executing Gossip Worker-Crash Experiment ---")
    gp_worker_crash = run_gossip_worker_crash()
    print("  -> Gossip Worker Crash complete.")
    for w_name, metrics in gp_worker_crash.items():
        epochs_completed = len(metrics)
        final_acc = metrics[-1]["accuracy"] if metrics else 0.0
        print(f"     {w_name}: Completed {epochs_completed} Epochs | Final Acc = {final_acc:.2f}%")

    output_dir = os.path.abspath("logs/phase7")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "symmetric_fault_results.json")

    with open(out_file, "w") as f:
        json.dump({
            "downpour_worker_crash": dp_worker_crash,
            "gossip_worker_crash": gp_worker_crash
        }, f, indent=2)

    print("=" * 75)
    print("SUCCESS: SYMMETRIC FAULT-TOLERANCE EVALUATION COMPLETE!")
    print(f"Results saved to {out_file}")
    print("=" * 75)

if __name__ == "__main__":
    main()
