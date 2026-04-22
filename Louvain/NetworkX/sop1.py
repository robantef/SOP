"""
SOP 1 (NetworkX) - Part B only
Run Louvain 30 times on the C-Town network to show non-deterministic outputs.
"""

from __future__ import annotations

import collections
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import wntr


def partition_signature(partition: list[set]) -> frozenset:
    """Return a canonical representation so partitions can be compared."""
    return frozenset(frozenset(community) for community in partition)

# Run Louvain for 30 times using the dataset
def run_ctown_experiment(inp_file: Path, num_runs: int = 30) -> tuple[list[int], list[float], int]:
    wn = wntr.network.WaterNetworkModel(str(inp_file))
    graph = nx.Graph(wn.to_graph())

    print(f"Loaded C-Town: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    community_counts: list[int] = []
    modularity_scores: list[float] = []
    partition_signatures: list[frozenset] = []

    for seed in range(num_runs):
        partition = nx.community.louvain_communities(graph, seed=seed)
        community_counts.append(len(partition))
        modularity_scores.append(nx.community.modularity(graph, partition))
        partition_signatures.append(partition_signature(partition))

    distinct_partitions = len(set(partition_signatures))
    return community_counts, modularity_scores, distinct_partitions

# For producing the figure
def plot_results(community_counts: list[int], modularity_scores: list[float], output_file: Path) -> None:
    """Save summary plots for community-count and modularity variance."""
    counter = collections.Counter(community_counts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"SOP 1 - C-Town Louvain Variance ({len(community_counts)} runs)",
        fontsize=12,
        fontweight="bold",
    )

    axes[0].bar(
        sorted(counter.keys()),
        [counter[k] for k in sorted(counter)],
        color="#e15759",
        edgecolor="#7b2021",
    )
    axes[0].set_xlabel("Number of Communities Detected")
    axes[0].set_ylabel(f"Frequency (out of {len(community_counts)} runs)")
    axes[0].set_title("Distribution of Community Counts")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(range(len(modularity_scores)), modularity_scores, "o-", color="#76b7b2", linewidth=1, markersize=4)
    axes[1].axhline(
        statistics.median(modularity_scores),
        color="navy",
        linestyle="--",
        linewidth=1.5,
        label=f"Median Q={statistics.median(modularity_scores):.4f}",
    )
    axes[1].set_xlabel("Run index")
    axes[1].set_ylabel("Modularity Q")
    axes[1].set_title("Modularity Score per Run")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")

#main
def main() -> None:
    num_runs = 30
    inp_file = Path(__file__).resolve().parents[2] / "GN Algorithm" / "CTOWN.INP"
    output_file = Path(__file__).with_name("sop1_ctown_nondeterminism.png")

    print("=" * 65)
    print("SOP 1 - Part B Only (C-Town Non-Determinism)")
    print("=" * 65)
    print(f"Running Louvain {num_runs} times on C-Town...\n")

    if not inp_file.exists():
        print(f"Could not find input file: {inp_file}")
        return

    community_counts, modularity_scores, distinct_partitions = run_ctown_experiment(inp_file, num_runs=num_runs)
    counter = collections.Counter(community_counts)

    print(f"Runs               : {num_runs}")
    print(f"Distinct partitions: {distinct_partitions}")
    print(f"Community range    : {min(community_counts)} - {max(community_counts)}")
    # print(f"Modularity range   : {min(modularity_scores):.4f} - {max(modularity_scores):.4f}")

    print("\nCommunity count distribution:")
    for k in sorted(counter):
        print(f"  {k:3d} communities: {counter[k]:2d} runs")

    plot_results(community_counts, modularity_scores, output_file)
    print(f"\nFigure saved: {output_file}")


if __name__ == "__main__":
    main()
