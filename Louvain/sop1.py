"""
SOP 1 – Node Ordering Sensitivity → Non-Deterministic Results
==============================================================
Problem Statement:
    The Louvain algorithm is sensitive to node ordering, leading to unstable
    and non-deterministic community detection results.

Demonstration:
    Part A – Synthetic medium-sized network:
        We construct a planted-partition (benchmark) graph with known ground-truth
        communities, then run Louvain 100 times with different random seeds (i.e.
        different node orderings). We measure:
          • How many distinct partitions are produced.
          • The distribution of the number of detected communities across runs.
          • The Normalised Mutual Information (NMI) between each run and the
            ground truth — lower NMI means the run diverged from the truth.

    Part B – C-Town water network:
        We run Louvain 30 times on the real-world C-Town network, recording
        the number of communities and modularity score for each run, and
        visualise the variance to confirm non-determinism on a real dataset.
"""

import networkx as nx
import wntr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import collections
import statistics
from sklearn.metrics import normalized_mutual_info_score

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def partition_to_labels(partition, node_list):
    """Convert a list-of-sets partition to a label array aligned with node_list."""
    label = {}
    for comm_id, comm in enumerate(partition):
        for node in comm:
            label[node] = comm_id
    return [label[n] for n in node_list]


def partition_signature(partition):
    """Canonical frozenset representation for equality comparison."""
    return frozenset(frozenset(c) for c in partition)


# ─────────────────────────────────────────────────────────────────────────────
# Part A – Planted-Partition Benchmark
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("SOP 1  –  Node-Ordering Sensitivity / Non-Determinism")
print("=" * 65)

# Build a planted-partition graph with WEAK community signal:
#   10 groups of 10 nodes, p_in=0.4 (sparse intra), p_out=0.15 (noisy inter)
#   This makes boundaries ambiguous so Louvain gets trapped in different local optima.
NUM_GROUPS  = 10
GROUP_SIZE  = 10
P_IN        = 0.4    # reduced from 0.7 — weaker internal cohesion
P_OUT       = 0.15   # increased from 0.05 — noisier inter-community edges
NUM_RUNS    = 100

rng = random.Random(0)
G_bench = nx.planted_partition_graph(NUM_GROUPS, GROUP_SIZE, P_IN, P_OUT, seed=rng.randint(0, 9999))
node_list = list(G_bench.nodes())

# Ground truth: nodes 0..9 → group 0, 10..19 → group 1, etc.
ground_truth_labels = [n // GROUP_SIZE for n in node_list]

print(f"\n[Part A]  Planted-Partition Benchmark: "
      f"{NUM_GROUPS} groups × {GROUP_SIZE} nodes, "
      f"p_in={P_IN}, p_out={P_OUT}")

all_signatures   = []
num_communities  = []
nmi_scores       = []
modularity_scores = []

for seed in range(NUM_RUNS):
    partition = nx.community.louvain_communities(G_bench, seed=seed)
    sig = partition_signature(partition)
    all_signatures.append(sig)
    num_communities.append(len(partition))
    labels = partition_to_labels(partition, node_list)
    nmi = normalized_mutual_info_score(ground_truth_labels, labels)
    nmi_scores.append(nmi)
    q = nx.community.modularity(G_bench, partition)
    modularity_scores.append(q)

distinct_partitions = len(set(all_signatures))
counter = collections.Counter(num_communities)

print(f"  Runs               : {NUM_RUNS}")
print(f"  Distinct partitions: {distinct_partitions}  "
      f"({'non-deterministic ⚠' if distinct_partitions > 1 else 'stable ✓'})")
print(f"  # Communities range: {min(num_communities)} – {max(num_communities)}  "
      f"(expected {NUM_GROUPS})")
print(f"  NMI range          : {min(nmi_scores):.4f} – {max(nmi_scores):.4f}  "
      f"(1.0 = perfect match with ground truth)")
print(f"  Median NMI         : {statistics.median(nmi_scores):.4f}")
print(f"  Modularity range   : {min(modularity_scores):.4f} – {max(modularity_scores):.4f}")

print(f"\n  Distribution of detected community counts across {NUM_RUNS} runs:")
for k in sorted(counter):
    bar = "█" * counter[k]
    print(f"    {k:3d} communities : {counter[k]:3d}× {bar}")

# ── Figure A ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("SOP 1 – Non-Determinism Due to Node Ordering Sensitivity",
             fontsize=13, fontweight="bold")

# Panel 1 – Histogram of community counts
axes[0].bar(sorted(counter.keys()), [counter[k] for k in sorted(counter)],
            color="#4f86c6", edgecolor="#1a3f6f")
axes[0].axvline(NUM_GROUPS, color="tomato", linestyle="--", linewidth=2,
                label=f"Ground truth ({NUM_GROUPS})")
axes[0].set_xlabel("Number of Communities Detected")
axes[0].set_ylabel("Frequency (out of 100 runs)")
axes[0].set_title("Distribution of Community Counts")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.4)

# Panel 2 – NMI scores across runs
axes[1].plot(range(NUM_RUNS), nmi_scores, color="#59a14f", linewidth=0.8, alpha=0.8)
axes[1].axhline(statistics.median(nmi_scores), color="purple", linestyle="--",
                linewidth=1.5, label=f"Median NMI={statistics.median(nmi_scores):.3f}")
axes[1].axhline(1.0, color="tomato", linestyle=":", linewidth=1.2, label="Perfect (1.0)")
axes[1].set_xlabel("Run (seed index)")
axes[1].set_ylabel("NMI vs Ground Truth")
axes[1].set_title("Partition Quality per Run")
axes[1].set_ylim(0, 1.05)
axes[1].legend()
axes[1].grid(alpha=0.3)

# Panel 3 – Three example network plots with different partitions
# Pick 3 runs that produced different numbers of communities (if possible)
seeds_to_show = []
seen_nc = set()
for seed, nc in enumerate(num_communities):
    if nc not in seen_nc:
        seeds_to_show.append(seed)
        seen_nc.add(nc)
    if len(seeds_to_show) == 3:
        break
while len(seeds_to_show) < 3:   # pad if not enough variety
    seeds_to_show.append(seeds_to_show[-1] + 1)

pos_bench = nx.spring_layout(G_bench, seed=42)
axes[2].axis("off")

# embed 3 small sub-axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
inner_axes = []
for row in range(3):
    iax = axes[2].inset_axes([0, 1 - (row + 1) / 3, 1, 1 / 3 - 0.02])
    inner_axes.append(iax)

palette3 = plt.cm.get_cmap("tab20", NUM_GROUPS + 5)
for iax, seed_s in zip(inner_axes, seeds_to_show):
    part_s = nx.community.louvain_communities(G_bench, seed=seed_s)
    nc_s   = len(part_s)
    clrs_s = []
    cm_s = {}
    for ci, comm in enumerate(part_s):
        for nd in comm:
            cm_s[nd] = palette3(ci)
    clrs_s = [cm_s[n] for n in node_list]
    nx.draw_networkx(
        G_bench, pos=pos_bench, ax=iax, with_labels=False,
        node_color=clrs_s, node_size=25,
        edge_color="#cccccc", width=0.4,
    )
    iax.set_title(f"seed={seed_s}, {nc_s} comms", fontsize=7)
    iax.axis("off")

axes[2].set_title("3 Runs — Different Partitions", fontsize=10, y=1.02)

plt.tight_layout()
fig.savefig("sop1_nondeterminism_benchmark.png", dpi=150, bbox_inches="tight")
print("\n  [Figure saved] sop1_nondeterminism_benchmark.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part B – C-Town Water Network
# ─────────────────────────────────────────────────────────────────────────────

INP_FILE = "..\\GN Algorithm\\CTOWN.INP"
NUM_RUNS_CTOWN = 30

print(f"\n[Part B]  C-Town Water Network  ({NUM_RUNS_CTOWN} independent runs)")
try:
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    G_ctown = nx.Graph(wn.to_graph())
    print(f"  Loaded C-Town: {G_ctown.number_of_nodes()} nodes, "
          f"{G_ctown.number_of_edges()} edges")

    ct_nc   = []
    ct_q    = []
    ct_sigs = []

    for seed in range(NUM_RUNS_CTOWN):
        part = nx.community.louvain_communities(G_ctown, seed=seed)
        ct_nc.append(len(part))
        ct_q.append(nx.community.modularity(G_ctown, part))
        ct_sigs.append(partition_signature(part))

    ct_distinct = len(set(ct_sigs))
    ct_counter  = collections.Counter(ct_nc)

    print(f"  Distinct partitions: {ct_distinct}  "
          f"({'non-deterministic ⚠' if ct_distinct > 1 else 'stable ✓'})")
    print(f"  # Communities range: {min(ct_nc)} – {max(ct_nc)}")
    print(f"  Modularity range   : {min(ct_q):.4f} – {max(ct_q):.4f}")

    print(f"\n  Community count distribution across {NUM_RUNS_CTOWN} runs:")
    for k in sorted(ct_counter):
        bar = "█" * ct_counter[k]
        print(f"    {k:3d} communities: {ct_counter[k]:2d}× {bar}")

    # ── Figure B ─────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle(
        f"SOP 1 – C-Town: Non-Deterministic Louvain Results ({NUM_RUNS_CTOWN} runs)",
        fontsize=12, fontweight="bold",
    )

    # Left – community count distribution
    axes2[0].bar(sorted(ct_counter.keys()),
                 [ct_counter[k] for k in sorted(ct_counter)],
                 color="#e15759", edgecolor="#7b2021")
    axes2[0].set_xlabel("Number of Communities Detected")
    axes2[0].set_ylabel(f"Frequency (out of {NUM_RUNS_CTOWN} runs)")
    axes2[0].set_title("Distribution of Community Counts\n(C-Town network)")
    axes2[0].grid(axis="y", alpha=0.3)

    # Right – modularity across runs
    axes2[1].plot(range(NUM_RUNS_CTOWN), ct_q, "o-", color="#76b7b2",
                  linewidth=1, markersize=4)
    axes2[1].axhline(statistics.median(ct_q), color="navy", linestyle="--",
                     linewidth=1.5,
                     label=f"Median Q={statistics.median(ct_q):.4f}")
    axes2[1].set_xlabel("Run index")
    axes2[1].set_ylabel("Modularity Q")
    axes2[1].set_title("Modularity Score per Run\n(C-Town network)")
    axes2[1].legend()
    axes2[1].grid(alpha=0.3)

    plt.tight_layout()
    fig2.savefig("sop1_ctown_nondeterminism.png", dpi=150, bbox_inches="tight")
    print("  [Figure saved] sop1_ctown_nondeterminism.png")

except FileNotFoundError:
    print(f"  ⚠  Could not find {INP_FILE}. Skipping C-Town Part B.")

print("\n[Done] SOP 1 complete.\n")
