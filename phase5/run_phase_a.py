"""
run_phase_a.py — Phase A Automated Experiment Suite:
1. Network Latency Simulation (0ms, 20ms, 50ms, 100ms)
2. Bandwidth Throttling Simulation (Unlimited, 100 Mbps, 10 Mbps, 1 Mbps)
3. Gossip Frequency Sweep (gossip_every = 1, 5, 10)
4. Peak Per-Worker Strain Analysis vs. Cluster Total Traffic
"""

import argparse
import json
import os
import subprocess
import sys
import time


def get_python_exe(root_dir):
    venv_py = os.path.join(root_dir, "downpour", "venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable


def run_cmd(cmd, cwd=None):
    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return process


def wait_processes(processes):
    for p in processes:
        p.wait()


def run_downpour_experiment(num_workers=2, epochs=3, latency_ms=0.0, bandwidth_mbps=0.0, log_dir="logs/phase5/downpour"):
    os.makedirs(log_dir, exist_ok=True)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py_exe = get_python_exe(root_dir)
    port = 50080
    
    # 1. Start Server
    server_cmd = [
        py_exe,
        os.path.join(root_dir, "downpour", "server.py"),
        "--port", str(port),
        "--lr", "0.01",
    ]
    server_proc = run_cmd(server_cmd)
    time.sleep(3)  # Wait for server bind

    # 2. Start Workers
    worker_procs = []
    start_time = time.time()
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
    total_time = time.time() - start_time
    server_proc.kill()

    # Aggregate worker logs
    worker_bytes = []
    final_accs = []
    final_losses = []
    for w in range(num_workers):
        log_file = os.path.join(log_dir, f"worker_{w}_metrics.json")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                data = json.load(f)
                if data:
                    last_epoch = data[-1]
                    worker_bytes.append(last_epoch.get("total_bytes", 0))
                    final_accs.append(last_epoch.get("accuracy", 0.0))
                    final_losses.append(last_epoch.get("loss", 0.0))

    total_cluster_bytes = sum(worker_bytes)
    # In Downpour, the PS process handles total cluster bytes (sent + recv for all workers)
    peak_node_bytes = total_cluster_bytes

    return {
        "system": "Downpour",
        "num_workers": num_workers,
        "epochs": epochs,
        "latency_ms": latency_ms,
        "bandwidth_mbps": bandwidth_mbps,
        "wall_clock_time": total_time,
        "total_cluster_bytes": total_cluster_bytes,
        "peak_node_bytes": peak_node_bytes,
        "mean_accuracy": sum(final_accs) / len(final_accs) if final_accs else 0.0,
        "mean_loss": sum(final_losses) / len(final_losses) if final_losses else 0.0,
    }


def run_gossip_experiment(num_workers=2, epochs=3, gossip_every=1, latency_ms=0.0, bandwidth_mbps=0.0, log_dir="logs/phase5/gossip"):
    os.makedirs(log_dir, exist_ok=True)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py_exe = get_python_exe(root_dir)
    base_port = 50100

    worker_procs = []
    start_time = time.time()
    for w in range(num_workers):
        w_cmd = [
            py_exe,
            os.path.join(root_dir, "gossip", "gossip_worker.py"),
            "--worker-id", str(w),
            "--num-workers", str(num_workers),
            "--base-port", str(base_port),
            "--epochs", str(epochs),
            "--batch-size", "64",
            "--gossip-every", str(gossip_every),
            "--latency-ms", str(latency_ms),
            "--bandwidth-mbps", str(bandwidth_mbps),
            "--data-dir", os.path.join(root_dir, "downpour", "data"),
            "--log-dir", log_dir,
        ]
        worker_procs.append(run_cmd(w_cmd))

    wait_processes(worker_procs)
    total_time = time.time() - start_time

    # Aggregate worker logs
    worker_bytes = []
    final_accs = []
    final_losses = []
    for w in range(num_workers):
        log_file = os.path.join(log_dir, f"worker_{w}_metrics.json")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                data = json.load(f)
                if data:
                    last_epoch = data[-1]
                    worker_bytes.append(last_epoch.get("total_bytes", 0))
                    final_accs.append(last_epoch.get("accuracy", 0.0))
                    final_losses.append(last_epoch.get("loss", 0.0))

    total_cluster_bytes = sum(worker_bytes)
    # In Gossip, the peak single-node load is max_i(bytes_i)
    peak_node_bytes = max(worker_bytes) if worker_bytes else 0

    return {
        "system": "Gossip",
        "num_workers": num_workers,
        "epochs": epochs,
        "gossip_every": gossip_every,
        "latency_ms": latency_ms,
        "bandwidth_mbps": bandwidth_mbps,
        "wall_clock_time": total_time,
        "total_cluster_bytes": total_cluster_bytes,
        "peak_node_bytes": peak_node_bytes,
        "mean_accuracy": sum(final_accs) / len(final_accs) if final_accs else 0.0,
        "mean_loss": sum(final_losses) / len(final_losses) if final_losses else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase A Experiments Runner")
    parser.add_argument("--test-run", action="store_true", help="Run short 1-epoch validation run")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per experiment")
    args = parser.parse_args()

    epochs = 1 if args.test_run else args.epochs
    out_dir = os.path.abspath("logs/phase5")
    os.makedirs(out_dir, exist_ok=True)

    print("=========================================================================")
    print(f"STARTING PHASE A EXPERIMENTAL SUITE (Epochs: {epochs})")
    print("=========================================================================")

    all_results = {
        "latency_experiments": [],
        "bandwidth_experiments": [],
        "gossip_frequency_experiments": [],
    }

    # 1. Latency Experiments (0, 20, 50, 100 ms)
    latencies = [0, 20] if args.test_run else [0, 20, 50, 100]
    print("\n--- 1. Running Network Latency Matrix ---")
    for lat in latencies:
        print(f"Testing Latency = {lat} ms...")
        res_dp = run_downpour_experiment(num_workers=2, epochs=epochs, latency_ms=lat, log_dir=os.path.join(out_dir, f"lat_dp_{lat}ms"))
        res_gp = run_gossip_experiment(num_workers=2, epochs=epochs, gossip_every=1, latency_ms=lat, log_dir=os.path.join(out_dir, f"lat_gp_{lat}ms"))
        all_results["latency_experiments"].extend([res_dp, res_gp])

    # 2. Bandwidth Experiments (Unrestricted, 100, 10, 1 Mbps)
    bandwidths = [0, 100] if args.test_run else [0, 100, 10, 1]
    print("\n--- 2. Running Bandwidth Throttling Matrix ---")
    for bw in bandwidths:
        bw_label = "unlimited" if bw == 0 else f"{bw}Mbps"
        print(f"Testing Bandwidth = {bw_label}...")
        res_dp = run_downpour_experiment(num_workers=2, epochs=epochs, bandwidth_mbps=bw, log_dir=os.path.join(out_dir, f"bw_dp_{bw_label}"))
        res_gp = run_gossip_experiment(num_workers=2, epochs=epochs, gossip_every=1, bandwidth_mbps=bw, log_dir=os.path.join(out_dir, f"bw_gp_{bw_label}"))
        all_results["bandwidth_experiments"].extend([res_dp, res_gp])

    # 3. Gossip Frequency Sweep (gossip_every = 1, 5, 10)
    gossip_intervals = [1, 5] if args.test_run else [1, 5, 10]
    print("\n--- 3. Running Gossip Frequency Sweep ---")
    for ge in gossip_intervals:
        print(f"Testing Gossip Every = {ge} batches...")
        res_gp = run_gossip_experiment(num_workers=2, epochs=epochs, gossip_every=ge, log_dir=os.path.join(out_dir, f"freq_gp_every{ge}"))
        all_results["gossip_frequency_experiments"].append(res_gp)

    summary_file = os.path.join(out_dir, "phase_a_results.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=========================================================================")
    print(f"SUCCESS: PHASE A EXPERIMENTAL SUITE COMPLETE!")
    print(f"Results saved to {summary_file}")
    print("=========================================================================")


if __name__ == "__main__":
    main()
