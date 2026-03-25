import networkx as nx
import wntr

# Load the CTOWN.INP file using wntr
inp_file = "..\\GN Algorithm\\CTOWN.INP"
wn = wntr.network.WaterNetworkModel(inp_file)
G = nx.Graph(wn.to_graph())

print(f"Loaded C-Town: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Compute Louvain communities
communities = nx.community.louvain_communities(G)

print(f"Found {len(communities)} Louvain communities:")
for i, c in enumerate(communities, 1):
    print(f"Community {i}: size={len(c)}")
