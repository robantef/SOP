import warnings
warnings.filterwarnings('ignore')  # Suppress WNTR curve warnings

import wntr
import networkx as nx
import matplotlib.pyplot as plt
import random

# 1. Load the real C-Town water network dataset
wn = wntr.network.WaterNetworkModel('CTOWN.INP')
G_original = nx.Graph(wn.to_graph())
print(f"C-Town loaded: {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")

# 2. Create an identical copy, but shuffle the internal edge ordering
#    This simulates how arbitrary input ordering affects tie-breaking
edges = list(G_original.edges())
random.seed(42)
random.shuffle(edges) # Shuffling the data array!

G_shuffled = nx.Graph()
G_shuffled.add_nodes_from(G_original.nodes())
G_shuffled.add_edges_from(edges)

# 3. Get the very first split (2 communities) for both graphs
partition_original = next(nx.community.girvan_newman(G_original))
partition_shuffled = next(nx.community.girvan_newman(G_shuffled))

# 4. Plotting them side-by-side
#    Use a fixed layout so both panels show the same node positions for direct visual comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
pos = nx.kamada_kawai_layout(G_original)  # Fixed layout for both panels

# Left Plot: Original edge ordering
color_map_orig = ['#ff9999' if node in partition_original[0] else '#99ccff'
                  for node in G_original.nodes()]
nx.draw(G_original, pos, ax=axes[0], node_color=color_map_orig,
        with_labels=False, node_size=40, edge_color='gray', linewidths=0.4)
axes[0].set_title("Partition A\n(Original Edge Data Order)", fontsize=13)

# Right Plot: Shuffled edge ordering
color_map_shuff = ['#ff9999' if node in partition_shuffled[0] else '#99ccff'
                   for node in G_shuffled.nodes()]
nx.draw(G_original, pos, ax=axes[1], node_color=color_map_shuff,
        with_labels=False, node_size=40, edge_color='gray', linewidths=0.4)
axes[1].set_title("Partition B\n(Shuffled Edge Data Order)", fontsize=13)

# Add a shared legend
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='#ff9999', label='Community 1'),
    mpatches.Patch(color='#99ccff', label='Community 2'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle("Figure 1.3: Algorithmic Instability in C-Town Due to Arbitrary Tie-Breaking",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()