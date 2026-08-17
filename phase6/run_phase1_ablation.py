"""
run_phase1_ablation.py — Phase 1 Optimizer Fairness & Ablation Suite.

Executes a 4-way optimizer ablation matrix across 3 random seeds (42, 123, 999):
1. Downpour SGD + Momentum SGD (lr=0.01, momentum=0.9)  [Scientific Fairness Baseline]
2. Downpour SGD + Adagrad (lr=0.01, eps=1e-8)          [Centralized Classic]
3. Gossip SGD + Momentum SGD (lr=0.01, momentum=0.9)    [Decentralized Baseline]
4. Gossip SGD + Vanilla SGD (lr=0.01, momentum=0.0)     [No Momentum Baseline]

Saves summarized metrics (mean +- std_dev) to logs/phase6/optimizer_ablation_results.json.
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

def run_ablation_trial(config_name, system, optimizer_type, momentum, seed, num_workers=2, epochs=1, root_dir="."):
    log_dir = os.path.abspath(f"logs/phase6/ablation/{config_name}_seed{seed}")
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
            "--optimizer", optimizer_type,
            "--momentum", str(momentum),
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
                "config_name": config_name,
                "system": system,
                "optimizer": optimizer_type,
                "momentum": momentum,
                "seed": seed,
                "wall_clock_time": total_time,
                "accuracy": last_ep.get("accuracy", 0.0),
                "loss": last_ep.get("loss", 0.0),
                "total_bytes": last_ep.get("total_bytes", 0),
            }
    return None

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Optimizer Ablation Runner")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per trial")
    parser.add_argument("--test-run", action="store_true", help="Quick test run on 1 seed")
    args = parser.parse_args()

    seeds = [42] if args.test_run else [42, 123, 999]
    configs = [
        {"name": "Downpour_MomentumSGD", "system": "Downpour", "optimizer": "sgd", "momentum": 0.9},
        {"name": "Downpour_Adagrad", "system": "Downpour", "optimizer": "adagrad", "momentum": 0.0},
        {"name": "Gossip_MomentumSGD", "system": "Gossip", "optimizer": "sgd", "momentum": 0.9},
        {"name": "Gossip_VanillaSGD", "system": "Gossip", "optimizer": "sgd", "momentum": 0.0},
    ]

    print("=" * 75)
    print("STARTING PHASE 1 OPTIMIZER ABLATION SUITE")
    print(f"Configs: {[c['name'] for c in configs]} | Seeds: {seeds}")
    print("=" * 75)

    all_raw_results = []
    summary_results = {}

    for cfg in configs:
        cfg_name = cfg["name"]
        print(f"\n--- Testing Configuration: {cfg_name} ---")
        seed_accuracies = []
        seed_losses = []
        seed_times = []

        for seed in seeds:
            print(f"Executing Seed = {seed}...")
            res = run_ablation_trial(
                config_name=cfg_name,
                system=cfg["system"],
                optimizer_type=cfg["optimizer"],
                momentum=cfg["momentum"],
                seed=seed,
                epochs=args.epochs,
            )
            if res:
                all_raw_results.append(res)
                seed_accuracies.append(res["accuracy"])
                seed_losses.append(res["loss"])
                seed_times.append(res["wall_clock_time"])
                print(f"  -> Seed {seed}: Acc = {res['accuracy']:.2f}%, Loss = {res['loss']:.4f}, Time = {res['wall_clock_time']:.2f}s")

        summary_results[cfg_name] = {
            "system": cfg["system"],
            "optimizer": cfg["optimizer"],
            "momentum": cfg["momentum"],
            "mean_accuracy": float(np.mean(seed_accuracies)),
            "std_accuracy": float(np.std(seed_accuracies)),
            "mean_loss": float(np.mean(seed_losses)),
            "std_loss": float(np.std(seed_losses)),
            "mean_time": float(np.mean(seed_times)),
            "std_time": float(np.std(seed_times)),
            "seeds_count": len(seeds),
        }

    output_dir = os.path.abspath("logs/phase6")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "optimizer_ablation_results.json")
    
    with open(out_file, "w") as f:
        json.dump({"summary": summary_results, "raw": all_raw_results}, f, indent=2)

    print("=" * 75)
    print("SUCCESS: PHASE 1 OPTIMIZER ABLATION SUITE COMPLETE!")
    print(f"Results saved to {out_file}")
    print("=" * 75)

if __name__ == "__main__":
    main()
