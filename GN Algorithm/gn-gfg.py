import networkx as nx
import matplotlib.pyplot as plt
import wntr

def plot(G, title="Graph"):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="lightblue",
        edge_color="gray",
        node_size=60,
        width=0.7
    )
    
    # Edge betweenness labels
    if G.number_of_edges() <= 50:
        edge_labels = {e: round(v, 2) for e, v in nx.edge_betweenness_centrality(G).items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title)
    plt.show()

def best_girvan_newman_partition(
    G,
    max_communities=50,
    show_plots=False,
    min_delta=1e-6,
    patience=1,
):
    if show_plots:
        plot(G, "Full Graph")

    gn_generator = nx.community.girvan_newman(G)
    best_partition = None
    best_modularity = float("-inf")
    no_improvement_steps = 0

    for step, communities in enumerate(gn_generator, 1):
        communities = tuple(sorted(c) for c in communities)
        n_communities = len(communities)
        modularity = nx.community.modularity(G, communities)

        print(
            f"Step {step}: communities={n_communities}, modularity={modularity:.6f}"
        )

        if modularity > best_modularity + min_delta:
            best_modularity = modularity
            best_partition = communities
            no_improvement_steps = 0
        else:
            no_improvement_steps += 1

        if no_improvement_steps >= patience:
            print(
                f"Stopping early: modularity did not improve for {patience} step(s)."
            )
            break

        if n_communities >= max_communities:
            print(f"Stopping at safety cap: max_communities={max_communities}")
            break

    return best_partition, best_modularity

inp_file = "CTOWN.INP"
wn = wntr.network.WaterNetworkModel(inp_file)
G = nx.Graph(wn.to_graph())

print(f"Loaded C-Town: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

communities, modularity = best_girvan_newman_partition(
    G,
    max_communities=50,
    show_plots=False,
    min_delta=1e-6,
    patience=1,
)

print(f"\nBest partition modularity: {modularity:.6f}")
print("Final Communities:")
for i, c in enumerate(communities, 1):
    print(f"Community {i}: size={len(c)}")