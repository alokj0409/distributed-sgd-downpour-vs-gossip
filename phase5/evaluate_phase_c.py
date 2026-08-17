"""
evaluate_phase_c.py — Publication-Grade Visualization Engine for Phase A & B:
Generates high-DPI (300 DPI) figures matching IEEE conference standards:
1. fig1_runtime_vs_latency.png (Network Latency Crossover Analysis)
2. fig2_gossip_frequency_tradeoff.png (Gossip Interval Trade-off: Runtime, Acc, Bytes)
3. fig3_scalability_speedup_efficiency.png (Speedup S(N) & Efficiency E(N) for N=2,4,8)
4. fig4_peak_vs_total_traffic.png (Peak Single-Node Traffic vs. Total Cluster Bytes)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication styling
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
})


def plot_latency_crossover(phase_a_data, out_dir):
    lat_data = phase_a_data.get("latency_experiments", [])
    if not lat_data:
        return

    dp_times = [d["wall_clock_time"] for d in lat_data if d["system"] == "Downpour"]
    gp_times = [d["wall_clock_time"] for d in lat_data if d["system"] == "Gossip"]
    latencies = [d["latency_ms"] for d in lat_data if d["system"] == "Downpour"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(latencies, dp_times, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Downpour SGD (Parameter Server)')
    ax.plot(latencies, gp_times, 's--', color='#ff7f0e', linewidth=2.5, markersize=8, label='Gossip SGD (P2P Mesh)')

    ax.set_xlabel('Simulated One-Way Network Latency (ms)')
    ax.set_ylabel('Wall-Clock Execution Time (s)')
    ax.set_title('Network Latency Sensitivity & Decentralization Crossover')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    out_file = os.path.join(out_dir, "fig1_runtime_vs_latency.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_file}")


def plot_gossip_frequency(phase_a_data, out_dir):
    freq_data = phase_a_data.get("gossip_frequency_experiments", [])
    if not freq_data:
        return

    intervals = [d["gossip_every"] for d in freq_data]
    times = [d["wall_clock_time"] for d in freq_data]
    bytes_gb = [d["total_cluster_bytes"] / 1e9 for d in freq_data]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color = '#1f77b4'
    ax1.set_xlabel('Gossip Interval (gossip_every N batches)')
    ax1.set_ylabel('Execution Time (seconds)', color=color)
    bars = ax1.bar(np.array(intervals) - 0.2, times, width=0.4, color=color, alpha=0.85, label='Wall-Clock Time')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#2ca02c'
    ax2.set_ylabel('Total Cluster Payload (GB)', color=color)
    lines = ax2.plot(intervals, bytes_gb, 'o--', color=color, linewidth=2.5, markersize=8, label='Network Payload (GB)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Gossip Frequency Sweep: Runtime vs. Network Payload')
    fig.tight_layout()

    out_file = os.path.join(out_dir, "fig2_gossip_frequency_tradeoff.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_file}")


def plot_scalability(phase_b_data, out_dir):
    scale_data = phase_b_data.get("scalability_results", [])
    if not scale_data:
        return

    dp_workers = [d["num_workers"] for d in scale_data if d["system"] == "Downpour"]
    dp_speedups = [d.get("speedup", 1.0) for d in scale_data if d["system"] == "Downpour"]
    dp_effs = [d.get("parallel_efficiency", 1.0) * 100 for d in scale_data if d["system"] == "Downpour"]

    gp_workers = [d["num_workers"] for d in scale_data if d["system"] == "Gossip"]
    gp_speedups = [d.get("speedup", 1.0) for d in scale_data if d["system"] == "Gossip"]
    gp_effs = [d.get("parallel_efficiency", 1.0) * 100 for d in scale_data if d["system"] == "Gossip"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Speedup S(N)
    ax1.plot(dp_workers, dp_speedups, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Downpour SGD')
    ax1.plot(gp_workers, gp_speedups, 's--', color='#ff7f0e', linewidth=2.5, markersize=8, label='Gossip SGD')
    ax1.plot([2, 4, 8], [1, 2, 4], 'k:', alpha=0.5, label='Ideal Linear Speedup')
    ax1.set_xlabel('Number of Workers (N)')
    ax1.set_ylabel('Speedup S(N)')
    ax1.set_title('Parallel Speedup S(N)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Parallel Efficiency E(N)
    ax2.plot(dp_workers, dp_effs, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Downpour SGD')
    ax2.plot(gp_workers, gp_effs, 's--', color='#ff7f0e', linewidth=2.5, markersize=8, label='Gossip SGD')
    ax2.set_xlabel('Number of Workers (N)')
    ax2.set_ylabel('Parallel Efficiency E(N) (%)')
    ax2.set_title('Parallel Scaling Efficiency E(N)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig3_scalability_speedup_efficiency.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_file}")


def plot_peak_traffic(phase_a_data, out_dir):
    lat_data = phase_a_data.get("latency_experiments", [])
    if not lat_data:
        return

    dp_item = [d for d in lat_data if d["system"] == "Downpour" and d["latency_ms"] == 0]
    gp_item = [d for d in lat_data if d["system"] == "Gossip" and d["latency_ms"] == 0]

    if not dp_item or not gp_item:
        return

    dp = dp_item[0]
    gp = gp_item[0]

    categories = ['Total Cluster Traffic (GB)', 'Peak Single-Node Traffic (GB)']
    dp_vals = [dp["total_cluster_bytes"] / 1e9, dp["peak_node_bytes"] / 1e9]
    gp_vals = [gp["total_cluster_bytes"] / 1e9, gp["peak_node_bytes"] / 1e9]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, dp_vals, width, label='Downpour SGD (PS Bottleneck)', color='#1f77b4')
    ax.bar(x + width/2, gp_vals, width, label='Gossip SGD (P2P Mesh)', color='#ff7f0e')

    ax.set_ylabel('Network Payload (GB)')
    ax.set_title('Parameter Server Bottleneck vs. Gossip Load Balancing')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.tight_layout()

    out_file = os.path.join(out_dir, "fig4_peak_vs_total_traffic.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_file}")


def main():
    logs_dir = os.path.abspath("logs/phase5")
    a_file = os.path.join(logs_dir, "phase_a_results.json")
    b_file = os.path.join(logs_dir, "phase_b_results.json")

    if os.path.exists(a_file):
        with open(a_file, "r") as f:
            a_data = json.load(f)
            plot_latency_crossover(a_data, logs_dir)
            plot_gossip_frequency(a_data, logs_dir)
            plot_peak_traffic(a_data, logs_dir)

    if os.path.exists(b_file):
        with open(b_file, "r") as f:
            b_data = json.load(f)
            plot_scalability(b_data, logs_dir)


if __name__ == "__main__":
    main()
