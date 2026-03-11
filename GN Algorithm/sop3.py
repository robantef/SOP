import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# Use a symmetric cycle graph — every edge has mathematically equal betweenness
# centrality by symmetry, guaranteeing GN must resolve arbitrary ties.
# This makes the instability provable and reproducible.
N = 16
G_original = nx.cycle_graph(N)
print(f"Using a {N}-node cycle graph (all edges have equal betweenness by symmetry)")

# Get the baseline partition
partition_original = next(nx.community.girvan_newman(G_original))
edges = list(G_original.edges())

# Search for a shuffle seed that produces a visibly different partition
partition_shuffled = None
G_shuffled = None
for seed in range(50):
    edges_shuf = edges[:]
    random.seed(seed)
    random.shuffle(edges_shuf)
    G_test = nx.Graph()
    G_test.add_nodes_from(range(N))
    G_test.add_edges_from(edges_shuf)
    p_test = next(nx.community.girvan_newman(G_test))
    orig_sets = frozenset(map(frozenset, partition_original))
    test_sets = frozenset(map(frozenset, p_test))
    if orig_sets != test_sets:
        print(f"Instability confirmed with seed={seed}")
        G_shuffled = G_test
        partition_shuffled = p_test
        break

if partition_shuffled is None:
    print("Warning: No differing partition found in 50 seeds.")
    partition_shuffled = partition_original
    G_shuffled = G_original

# Report the two partitions
comm_a_orig = sorted(partition_original[0])
comm_b_orig = sorted(partition_original[1])
comm_a_shuf = sorted(partition_shuffled[0])
comm_b_shuf = sorted(partition_shuffled[1])
print(f"Partition A: {comm_a_orig} | {comm_b_orig}")
print(f"Partition B: {comm_a_shuf} | {comm_b_shuf}")

# Normalize color assignment: the community containing node 1 is always red.
# This fixes the reference point so visual differences are directly readable.
def make_colors(partition, n):
    anchor_comm = next(c for c in partition if 1 in c)
    return ['#e74c3c' if node in anchor_comm else '#3498db' for node in range(n)]

color_orig = make_colors(partition_original, N)
color_shuf = make_colors(partition_shuffled, N)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

nx.draw_circular(G_original, ax=axes[0], node_color=color_orig, with_labels=True,
                 node_size=700, font_weight='bold', edge_color='gray',
                 width=2, font_color='white')
axes[0].set_title(
    f"Partition A — Original Edge Order\n"
    f"Split: {len(comm_a_orig)} nodes | {len(comm_b_orig)} nodes",
    fontsize=12)

nx.draw_circular(G_original, ax=axes[1], node_color=color_shuf, with_labels=True,
                 node_size=700, font_weight='bold', edge_color='gray',
                 width=2, font_color='white')
axes[1].set_title(
    f"Partition B — Shuffled Edge Order\n"
    f"Split: {len(comm_a_shuf)} nodes | {len(comm_b_shuf)} nodes",
    fontsize=12)

legend_handles = [
    mpatches.Patch(color='#e74c3c', label='Community 1 (contains node 1)'),
    mpatches.Patch(color='#3498db', label='Community 2'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=11)

fig.suptitle(
    f"Figure 1.3: Algorithmic Instability Due to Arbitrary Tie-Breaking\n"
    f"(Symmetric {N}-Node Cycle — All Edges Have Equal Betweenness by Symmetry)",
    fontsize=13, fontweight='bold', y=1.02)
plt.subplots_adjust(bottom=0.12, top=0.88)
fig.savefig('figure_1_3_instability.png', dpi=150, bbox_inches='tight')
plt.show()