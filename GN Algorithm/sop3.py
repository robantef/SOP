import networkx as nx
import matplotlib.pyplot as plt
import random

# 1. Create a symmetrical grid graph where ties in betweenness will occur
G_original = nx.grid_2d_graph(3, 3)
G_original = nx.convert_node_labels_to_integers(G_original)

# 2. Create an identical graph, but shuffle the internal order of the edges
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
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pos = nx.spring_layout(G_original, seed=10) # Keep layout identical for visual comparison

# Left Plot: Original Order
color_map_orig = ['#ff9999' if node in partition_original[0] else '#99ccff' for node in G_original.nodes()]
nx.draw(G_original, pos, ax=axes[0], node_color=color_map_orig, with_labels=True, node_size=800, font_weight="bold")
axes[0].set_title("Partition A (Original Data Order)", fontsize=14)

# Right Plot: Shuffled Order
color_map_shuff = ['#ff9999' if node in partition_shuffled[0] else '#99ccff' for node in G_shuffled.nodes()]
nx.draw(G_original, pos, ax=axes[1], node_color=color_map_shuff, with_labels=True, node_size=800, font_weight="bold")
axes[1].set_title("Partition B (Shuffled Edge Data Order)", fontsize=14)

plt.suptitle("Figure 1.3: Algorithmic Instability Due to Arbitrary Tie-Breaking", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()