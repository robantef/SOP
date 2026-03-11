import networkx as nx
import matplotlib.pyplot as plt

# Load standard benchmark dataset
G = nx.karate_club_graph()
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
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42) # Seed for consistent layout
nx.draw(G, pos, node_color=color_map, with_labels=True, 
        node_size=600, edge_color="gray", font_weight="bold")

plt.title("Figure 1.2: Premature Fragmentation (Red Node is an Isolated Singleton)", fontsize=14)
plt.show()