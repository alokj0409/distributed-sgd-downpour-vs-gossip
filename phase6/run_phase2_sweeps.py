"""
run_phase2_sweeps.py — Phase 2 Fine-Grained Latency, Bandwidth & Convergence Sweeps.

Executes:
1. Fine-Grained Latency Sweep: 8 points (0, 10, 20, 30, 40, 50, 75, 100 ms)
2. Multi-Point Bandwidth Matrix: 5 points (unlimited, 500, 250, 100, 50 Mbps)

Saves summarized metrics to logs/phase6/fine_sweeps_results.json.
"""

import os
import sys
import json
import time
import subprocess
import argparse
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

def run_simulated_trial(system, latency_ms=0, bandwidth_mbps=0, num_workers=2, epochs=1, root_dir="."):
    tag = f"lat_{latency_ms}ms_bw_{bandwidth_mbps}Mbps"
    log_dir = os.path.abspath(f"logs/phase6/sweeps/{system}_{tag}")
    os.makedirs(log_dir, exist_ok=True)
    py_exe = get_python_exe(root_dir)

    start_time = time.time()
    if system == "Downpour":
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
                "--latency-ms", str(latency_ms),
                "--bandwidth-mbps", str(bandwidth_mbps),
                "--data-dir", os.path.join(root_dir, "downpour", "data"),
                "--log-dir", log_dir,
            ]
            worker_procs.append(run_cmd(w_cmd))

        wait_processes(worker_procs)
        server_proc.kill()

    else:  # Gossip
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
                "--latency-ms", str(latency_ms),
                "--bandwidth-mbps", str(bandwidth_mbps),
                "--data-dir", os.path.join(root_dir, "downpour", "data"),
                "--log-dir", log_dir,
            ]
            worker_procs.append(run_cmd(w_cmd))

        wait_processes(worker_procs)

    total_time = time.time() - start_time

    # Parse worker 0 metrics
    w0_file = os.path.join(log_dir, "worker_0_metrics.json")
    if os.path.exists(w0_file):
        with open(w0_file, "r") as f:
            data = json.load(f)
            last_ep = data[-1]
            return {
                "system": system,
                "latency_ms": latency_ms,
                "bandwidth_mbps": bandwidth_mbps,
                "wall_clock_time": total_time,
                "accuracy": last_ep.get("accuracy", 0.0),
                "loss": last_ep.get("loss", 0.0),
                "total_bytes": last_ep.get("total_bytes", 0),
                "throughput": last_ep.get("throughput_samples_per_sec", 0.0),
            }
    return None

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Fine-Grained Sweeps Runner")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per trial")
    parser.add_argument("--test-run", action="store_true", help="Quick test run on subset of points")
    args = parser.parse_args()

    if args.test_run:
        latencies = [0, 20, 50, 100]
        bandwidths = [0, 250, 100]
    else:
        latencies = [0, 10, 20, 30, 40, 50, 75, 100]
        bandwidths = [0, 500, 250, 100, 50]

    print("=" * 75)
    print("STARTING PHASE 2 FINE-GRAINED SWEEPS SUITE")
    print(f"Latencies: {latencies} ms | Bandwidths: {bandwidths} Mbps")
    print("=" * 75)

    latency_results = []
    bandwidth_results = []

    # 1. Fine-Grained Latency Sweep
    print("\n--- 1. Running Fine-Grained Latency Sweep ---")
    for lat in latencies:
        for sys_name in ["Downpour", "Gossip"]:
            print(f"Testing System = {sys_name}, Latency = {lat} ms...")
            res = run_simulated_trial(system=sys_name, latency_ms=lat, bandwidth_mbps=0, epochs=args.epochs)
            if res:
                latency_results.append(res)
                print(f"  -> {sys_name} ({lat}ms): Time = {res['wall_clock_time']:.2f}s, Throughput = {res['throughput']:.1f} samp/s")

    # 2. Multi-Point Bandwidth Throttling Sweep
    print("\n--- 2. Running Multi-Point Bandwidth Throttling Sweep ---")
    for bw in bandwidths:
        for sys_name in ["Downpour", "Gossip"]:
            print(f"Testing System = {sys_name}, Bandwidth = {bw} Mbps...")
            res = run_simulated_trial(system=sys_name, latency_ms=0, bandwidth_mbps=bw, epochs=args.epochs)
            if res:
                bandwidth_results.append(res)
                print(f"  -> {sys_name} ({bw}Mbps): Time = {res['wall_clock_time']:.2f}s, Throughput = {res['throughput']:.1f} samp/s")

    output_dir = os.path.abspath("logs/phase6")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fine_sweeps_results.json")

    with open(out_file, "w") as f:
        json.dump({"latency_sweeps": latency_results, "bandwidth_sweeps": bandwidth_results}, f, indent=2)

    print("=" * 75)
    print("SUCCESS: PHASE 2 FINE-GRAINED SWEEPS COMPLETE!")
    print(f"Results saved to {out_file}")
    print("=" * 75)

if __name__ == "__main__":
    main()
