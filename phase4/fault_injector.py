"""
fault_injector.py — Synthetic Fault & Latency Injector for Distributed SGD.

Provides latency emulation (simulating network delay or slow compute stragglers)
and process fault injection (killing worker or parameter server processes mid-training).
"""

import os
import signal
import sys
import time
import random


class FaultInjector:
    def __init__(self, latency_ms: float = 0.0, drop_rate: float = 0.0):
        self.latency_sec = latency_ms / 1000.0
        self.drop_rate = drop_rate

    def inject_latency(self):
        """Simulate network or compute straggler delay."""
        if self.latency_sec > 0:
            time.sleep(self.latency_sec)

    def should_drop_packet(self) -> bool:
        """Simulate network packet loss."""
        return random.random() < self.drop_rate


def terminate_process_by_pid(pid: int):
    """Forcefully terminate a process given its PID (simulating hardware crash)."""
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGKILL)
        print(f"[FAULT INJECTOR] Successfully killed process PID {pid}")
        return True
    except Exception as e:
        print(f"[FAULT INJECTOR] Failed to kill process PID {pid}: {e}")
        return False
