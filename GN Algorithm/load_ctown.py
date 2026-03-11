import wntr
import networkx as nx

# 1. Load the C-Town .inp file (CTOWN.INP is in the same folder as this script)
inp_file = 'CTOWN.INP'
wn = wntr.network.WaterNetworkModel(inp_file)

# 2. Extract the NetworkX graph from the hydraulic model
# WNTR returns a directed multigraph by default (accounts for flow direction & parallel pipes)
multi_graph = wn.to_graph()

# 3. Simplify to a plain undirected graph for topological analysis
# This removes flow direction and parallel edges, leaving only the network structure
G_ctown = nx.Graph(multi_graph)

# 4. Verify the result
print(f"Successfully loaded C-Town!")
print(f"Original Hydraulic Model : {wn.num_nodes} nodes, {wn.num_links} pipes/links")
print(f"Abstracted Graph         : {G_ctown.number_of_nodes()} nodes, {G_ctown.number_of_edges()} edges")
print(f"Is connected             : {nx.is_connected(G_ctown)}")
