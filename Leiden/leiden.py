import wntr
import networkx as nx
import igraph as ig
import leidenalg

# Load CTOWN.INP using wntr
inp_file = "..\\GN Algorithm\\CTOWN.INP"
wn = wntr.network.WaterNetworkModel(inp_file)
G_nx = nx.Graph(wn.to_graph())

print(f"Loaded C-Town: {G_nx.number_of_nodes()} nodes, {G_nx.number_of_edges()} edges")

# Convert NetworkX graph to igraph
G_ig = ig.Graph.TupleList(G_nx.edges(), directed=False)

# Run Leiden algorithm
partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=42)

print(f"Found {len(partition)} Leiden communities:")
for i, community in enumerate(partition, 1):
	print(f"Community {i}: size={len(community)}")

