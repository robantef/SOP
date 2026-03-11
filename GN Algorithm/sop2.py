import warnings
warnings.filterwarnings('ignore')  # Suppress WNTR curve warnings

import wntr
import networkx as nx
import matplotlib.pyplot as plt

# Load the real C-Town water network dataset
wn = wntr.network.WaterNetworkModel('CTOWN.INP')
G = nx.Graph(wn.to_graph())
print(f"C-Town loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print("Running Girvan-Newman on C-Town to find first singleton fragmentation...")

gn_generator = nx.community.girvan_newman(G)

# Iterate through the splits to find the first time a single node is isolated
singleton_partition = None
for partition in gn_generator:
    # Check if any community in the current partition has exactly 1 node
    if any(len(community) == 1 for community in partition):
        singleton_partition = partition
        break

# Map colors to the communities
color_map = []
for node in G.nodes():
    for i, comm in enumerate(singleton_partition):
        if node in comm:
            # Highlight the singleton in red, others in light colors
            if len(comm) == 1:
                color_map.append('red') 
            else:
                color_map.append('lightblue' if i % 2 == 0 else 'lightgreen')

# Plotting the graph
# Use kamada_kawai layout for cleaner infrastructure network visualization
plt.figure(figsize=(12, 8))
pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos, node_color=color_map, with_labels=False,
        node_size=60, edge_color="gray", linewidths=0.5)

# Add a legend to explain the colors
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='red',       label='Isolated Singleton Node (Prematurely Split Off)'),
    mpatches.Patch(color='lightblue', label='Community A'),
    mpatches.Patch(color='lightgreen',label='Community B'),
]
plt.legend(handles=legend_handles, loc='upper left', fontsize=10)
plt.title("Figure 1.2: Premature Fragmentation in C-Town Water Network\n"
          "(Red node is an isolated singleton)", fontsize=14)
plt.show()