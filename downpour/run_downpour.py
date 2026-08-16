"""
run_downpour.py — Launcher script for the Downpour SGD prototype.

Starts 1 parameter server + N workers as subprocesses, monitors their output,
and collects training logs when complete.
"""

import argparse
import os
import signal
import subprocess
import sys
import time


# Use the venv python by default
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(SCRIPT_DIR, "venv", "Scripts", "python.exe")

# Fallback to current python if venv doesn't exist
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable


def start_server(port: int, lr: float) -> subprocess.Popen:
    """Start the parameter server as a subprocess."""
    cmd = [
        VENV_PYTHON,
        os.path.join(SCRIPT_DIR, "server.py"),
        "--port", str(port),
        "--lr", str(lr),
    ]
    print(f"[LAUNCHER] Starting parameter server on port {port}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )
    return proc


def start_worker(
    worker_id: int,
    server_address: str,
    num_workers: int,
    epochs: int,
    batch_size: int,
    pull_every: int,
    data_dir: str,
    log_dir: str,
) -> subprocess.Popen:
    """Start a worker as a subprocess."""
    cmd = [
        VENV_PYTHON,
        os.path.join(SCRIPT_DIR, "worker.py"),
        "--worker-id", str(worker_id),
        "--server-address", server_address,
        "--num-workers", str(num_workers),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--pull-every", str(pull_every),
        "--data-dir", data_dir,
        "--log-dir", log_dir,
    ]
    print(f"[LAUNCHER] Starting worker {worker_id}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def stream_output(proc: subprocess.Popen, label: str):
    """Read and print output from a subprocess line by line."""
    for line in proc.stdout:
        print(f"{line}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Downpour SGD Launcher")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--port", type=int, default=50051, help="Parameter server port")
    parser.add_argument("--lr", type=float, default=0.01, help="Adagrad learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per worker")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--pull-every", type=int, default=1, help="Pull weights every N batches")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 data directory")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Training log directory")
    args = parser.parse_args()

    server_address = f"localhost:{args.port}"
    processes = []

    try:
        # 1. Start parameter server
        server_proc = start_server(args.port, args.lr)
        processes.append(("server", server_proc))

        # Give server time to start
        print("[LAUNCHER] Waiting for parameter server to start...")
        time.sleep(3)

        # Check if server started successfully
        if server_proc.poll() is not None:
            print("[LAUNCHER] ERROR: Parameter server failed to start!")
            output = server_proc.stdout.read()
            print(output)
            return

        # 2. Start workers
        worker_procs = []
        for i in range(args.num_workers):
            worker_proc = start_worker(
                worker_id=i,
                server_address=server_address,
                num_workers=args.num_workers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                pull_every=args.pull_every,
                data_dir=args.data_dir,
                log_dir=args.log_dir,
            )
            worker_procs.append(worker_proc)
            processes.append((f"worker-{i}", worker_proc))

        print(f"[LAUNCHER] All {args.num_workers} workers started. Training in progress...")
        print("=" * 70)

        # 3. Stream output from all processes
        import threading

        threads = []
        for label, proc in processes:
            t = threading.Thread(target=stream_output, args=(proc, label), daemon=True)
            t.start()
            threads.append(t)

        # 4. Wait for all workers to finish
        for worker_proc in worker_procs:
            worker_proc.wait()

        print("=" * 70)
        print("[LAUNCHER] All workers finished training!")

        # 5. Run evaluation
        print("[LAUNCHER] Running final evaluation...")
        eval_cmd = [
            VENV_PYTHON,
            os.path.join(SCRIPT_DIR, "evaluate.py"),
            "--server-address", server_address,
            "--data-dir", args.data_dir,
            "--log-dir", args.log_dir,
        ]
        eval_proc = subprocess.run(eval_cmd, capture_output=False, text=True)

    except KeyboardInterrupt:
        print("\n[LAUNCHER] Interrupted! Shutting down...")

    finally:
        # Terminate all processes
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
