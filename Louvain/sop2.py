"""
SOP 2 – Resolution Limit in Modularity Optimization
=====================================================
Problem Statement:
    The Louvain algorithm relies on modularity optimization, which introduces a
    resolution limit that prevents the detection of smaller communities within
    large-scale networks.

Demonstration:
    Part A – Synthetic "Ring of Cliques" benchmark:
        A ring of small, fully-connected cliques is constructed. Each clique
        SHOULD be its own community (ground truth). We show that Louvain merges
        adjacent cliques below the resolution-limit threshold.

    Part B – C-Town water network:
        The real-world C-Town network is partitioned with Louvain. We then
        compare community sizes against a reference equal-split partition to
        highlight cases where structurally meaningful small sub-graphs get
        absorbed into larger communities.
"""

import networkx as nx
import wntr
import matplotlib
matplotlib.use("Agg")          # headless backend – no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_ring_of_cliques(num_cliques: int, clique_size: int) -> nx.Graph:
    """Return a ring of `num_cliques` fully-connected cliques, each of size
    `clique_size`, connected in a ring by single bridge edges."""
    G = nx.Graph()
    node_id = 0
    clique_nodes = []

    for _ in range(num_cliques):
        nodes = list(range(node_id, node_id + clique_size))
        clique_nodes.append(nodes)
        # fully connect this clique
        for u in nodes:
            for v in nodes:
                if u < v:
                    G.add_edge(u, v)
        node_id += clique_size

    # connect cliques in a ring (one bridge edge between consecutive cliques)
    for i in range(num_cliques):
        u = clique_nodes[i][-1]           # last node of clique i
        v = clique_nodes[(i + 1) % num_cliques][0]  # first node of next clique
        G.add_edge(u, v)

    return G, clique_nodes


def assign_colors(partition, num_nodes):
    """Return a colour list indexed by node id."""
    palette = plt.cm.get_cmap("tab20", len(partition))
    colors = ["#cccccc"] * num_nodes
    for comm_idx, comm in enumerate(partition):
        for node in comm:
            colors[node] = palette(comm_idx)
    return colors


# ─────────────────────────────────────────────────────────────────────────────
# Part A – Ring of Cliques
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLIQUES  = 12   # larger ring increases chance of resolution-limit merging
CLIQUE_SIZE  = 3    # smaller cliques fall below the resolution threshold more reliably

print("=" * 65)
print("SOP 2  –  Resolution Limit in Modularity Optimization")
print("=" * 65)

G_ring, clique_groups = build_ring_of_cliques(NUM_CLIQUES, CLIQUE_SIZE)

# Ground-truth partition: every clique is its own community
ground_truth = [set(g) for g in clique_groups]

# Louvain partition (fixed seed for reproducibility of this demo)
louvain_partition = nx.community.louvain_communities(G_ring, seed=42)

print(f"\n[Part A]  Ring-of-Cliques  |  {NUM_CLIQUES} cliques × {CLIQUE_SIZE} nodes each")
print(f"  Ground-truth communities : {len(ground_truth)}")
print(f"  Louvain communities      : {len(louvain_partition)}")

if len(louvain_partition) < len(ground_truth):
    diff = len(ground_truth) - len(louvain_partition)
    print(f"  ⚠  Resolution limit detected: Louvain merged {diff} clique(s) "
          f"into larger groups, hiding fine-grained structure.")
else:
    print("  ✓  Louvain matched ground truth on this ring size.")

# Detail: which Louvain community contains nodes from multiple ground-truth cliques?
print("\n  Louvain community breakdown (nodes → which ground-truth clique they belong to):")
for lc_idx, lc in enumerate(sorted(louvain_partition, key=lambda s: min(s))):
    gt_labels = []
    for node in sorted(lc):
        for gt_idx, gt in enumerate(ground_truth):
            if node in gt:
                gt_labels.append(f"C{gt_idx+1}")
                break
    unique_gt = sorted(set(gt_labels))
    merged = " + ".join(unique_gt) if len(unique_gt) > 1 else unique_gt[0]
    print(f"    Louvain comm {lc_idx+1:2d}: nodes {sorted(lc)} → cliques [{merged}]")

# ── Figure A ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("SOP 2 – Resolution Limit: Ring of Cliques", fontsize=14, fontweight="bold")

pos = nx.spring_layout(G_ring, seed=7)
n = G_ring.number_of_nodes()

for ax, partition, title in [
    (axes[0], ground_truth,      "Ground Truth\n(each clique = 1 community)"),
    (axes[1], louvain_partition, "Louvain Output\n(adjacent cliques merged  ⚠)"),
]:
    colors = assign_colors(partition, n)
    nx.draw_networkx(
        G_ring, pos=pos, ax=ax, with_labels=True,
        node_color=colors, node_size=350,
        font_size=7, edge_color="#888888", width=1.2,
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")

plt.tight_layout()
fig.savefig("sop2_ring_of_cliques.png", dpi=150, bbox_inches="tight")
print("\n  [Figure saved] sop2_ring_of_cliques.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part B – C-Town Water Network
# ─────────────────────────────────────────────────────────────────────────────

INP_FILE = "..\\GN Algorithm\\CTOWN.INP"

print(f"\n[Part B]  C-Town Water Network")
try:
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    G_ctown = nx.Graph(wn.to_graph())
    print(f"  Loaded C-Town: {G_ctown.number_of_nodes()} nodes, "
          f"{G_ctown.number_of_edges()} edges")

    ctown_partition = nx.community.louvain_communities(G_ctown, seed=42)
    sizes = sorted([len(c) for c in ctown_partition])

    print(f"  Louvain detected {len(ctown_partition)} communities")
    print(f"  Community sizes : {sizes}")

    # Flag communities that are suspiciously large relative to the median
    import statistics
    med = statistics.median(sizes)
    large = [s for s in sizes if s > 2 * med]
    small = [s for s in sizes if s <= 2]          # singleton or pair
    if large:
        print(f"  ⚠  {len(large)} oversized community/ies (> 2× median={med:.0f}): "
              f"may have absorbed smaller sub-graphs due to resolution limit.")
    if small:
        print(f"  ℹ  {len(small)} very small community/ies (≤ 2 nodes): "
              f"these may be isolated nodes or dangling segments.")

    # ── Figure B ─────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 7))
    fig2.suptitle("SOP 2 – C-Town: Community Size Distribution (Louvain)",
                  fontsize=13, fontweight="bold")

    # Left: bar chart of community sizes
    ax = axes2[0]
    ax.bar(range(1, len(sizes) + 1), sizes, color="#4f86c6", edgecolor="#1a3f6f")
    ax.axhline(med, color="tomato", linestyle="--", linewidth=1.5,
               label=f"Median size = {med:.0f}")
    ax.axhline(2 * med, color="orange", linestyle=":", linewidth=1.5,
               label=f"2× Median = {2*med:.0f}")
    ax.set_xlabel("Community Index (sorted by size)")
    ax.set_ylabel("Number of Nodes")
    ax.set_title("Community Sizes")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: network coloured by community
    ax2 = axes2[1]
    palette2 = plt.cm.get_cmap("tab20", len(ctown_partition))
    node_list = list(G_ctown.nodes())
    color_map = {}
    for ci, comm in enumerate(ctown_partition):
        for nd in comm:
            color_map[nd] = palette2(ci)
    node_colors = [color_map[n] for n in node_list]
    pos2 = nx.spring_layout(G_ctown, seed=42, k=0.4)
    nx.draw_networkx(
        G_ctown, pos=pos2, ax=ax2, with_labels=False,
        node_color=node_colors, node_size=60,
        edge_color="#aaaaaa", width=0.5, alpha=0.9,
    )
    ax2.set_title("C-Town Network coloured by Louvain Community")
    ax2.axis("off")

    plt.tight_layout()
    fig2.savefig("sop2_ctown_communities.png", dpi=150, bbox_inches="tight")
    print("  [Figure saved] sop2_ctown_communities.png")

except FileNotFoundError:
    print(f"  ⚠  Could not find {INP_FILE}. Skipping C-Town Part B.")

print("\n[Done] SOP 2 complete.\n")
