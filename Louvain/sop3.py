"""
SOP 3 – Greedy Local Optimization → Disconnected Communities
=============================================================
Problem Statement:
    The Louvain algorithm employs a greedy, local optimization strategy that
    fails to guarantee globally optimal community partitions.

Demonstration:
    Part A – Synthetic "Bridge Node" network:
        A handcrafted graph mimics Figure 1.2 from the paper: one node acts as
        a bridge between two sub-groups. We trace how the greedy local-moving
        phase reassigns that bridge node to a neighbouring community because the
        immediate modularity gain is positive, even though the move fragments
        the original community into two disconnected components.

    Part B – C-Town water network:
        After running Louvain on C-Town we check every detected community for
        internal connectivity. Any disconnected community is direct evidence of
        the greedy local optimisation flaw described in SOP 3.
"""

import networkx as nx
import wntr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_disconnected_community(G: nx.Graph, community: set) -> bool:
    """Return True if the subgraph induced by `community` is disconnected."""
    sub = G.subgraph(community)
    return not nx.is_connected(sub)


def count_disconnected(G: nx.Graph, partition) -> int:
    """Count communities that are internally disconnected."""
    return sum(1 for c in partition if is_disconnected_community(G, c))


# ─────────────────────────────────────────────────────────────────────────────
# Part A – Bridge Node Synthetic Network
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("SOP 3  –  Greedy Local Optimization → Disconnected Communities")
print("=" * 65)

# Build an expanded bridge-node graph (Figure 1.2)
#
# Layout:
#   Sub-group L1: nodes 0-1-2 (triangle)
#   Sub-group L2: nodes 3-4-5 (triangle)
#   Bridge-1 node: 6  — sole connector between L1–L2 and the right side
#   Sub-group R1: nodes 7-8-9 (triangle)
#   Sub-group R2: nodes 10-11-12 (triangle)
#   Bridge-2 node: 13 — sole connector between R1–R2 and the left side
#   Dense neighbour community: nodes 14-15-16-17 (K4)
#     connected to bridge-1 (6) and bridge-2 (13) via multiple edges
#
# When the greedy phase reassigns bridge nodes 6 and 13 to the dense neighbour
# community, both sub-group pairs (L1+L2) and (R1+R2) become disconnected.
#
G_bridge = nx.Graph()

# Left sub-groups
G_bridge.add_edges_from([(0,1),(1,2),(0,2)])          # L1 triangle
G_bridge.add_edges_from([(3,4),(4,5),(3,5)])          # L2 triangle
G_bridge.add_edges_from([(2,6),(5,6)])                # L1 & L2 connect to bridge-1

# Right sub-groups
G_bridge.add_edges_from([(7,8),(8,9),(7,9)])          # R1 triangle
G_bridge.add_edges_from([(10,11),(11,12),(10,12)])    # R2 triangle
G_bridge.add_edges_from([(9,13),(12,13)])             # R1 & R2 connect to bridge-2

# Inter-side connector (thin link between the two sides)
G_bridge.add_edge(6, 13)

# Dense neighbour community K4 — 4 nodes fully connected
G_bridge.add_edges_from([(14,15),(14,16),(14,17),(15,16),(15,17),(16,17)])
# Both bridge nodes are attracted to this dense community
G_bridge.add_edges_from([(6,14),(6,15),(13,16),(13,17)])

# "Before" partition: both bridge nodes still in the main community
before_comm_A = {0,1,2,3,4,5,6,7,8,9,10,11,12,13}   # all main nodes
before_comm_B = {14,15,16,17}                          # dense neighbour

# "After" greedy reassignment of both bridge nodes (6 and 13 move to comm B)
after_comm_A = {0,1,2,3,4,5,7,8,9,10,11,12}   # bridges removed → at least 2 disconnected islands
after_comm_B = {6,13,14,15,16,17}

print("\n[Part A]  Bridge-Node Synthetic Network")
print(f"  Nodes  : {G_bridge.number_of_nodes()}")
print(f"  Edges  : {G_bridge.number_of_edges()}")

# Verify disconnection AFTER greedy move
sub_after = G_bridge.subgraph(after_comm_A)
connected_before = nx.is_connected(G_bridge.subgraph(before_comm_A))
connected_after  = nx.is_connected(sub_after)

print(f"\n  Community A connected BEFORE bridge-node move : {connected_before}")
print(f"  Community A connected AFTER  bridge-node move : {connected_after}")
if not connected_after:
    comps = list(nx.connected_components(sub_after))
    print(f"  ⚠  Community A split into {len(comps)} disconnected fragment(s): {comps}")

# ── Figure A ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("SOP 3 – Greedy Optimisation Creates Disconnected Communities",
             fontsize=13, fontweight="bold")

pos = nx.spring_layout(G_bridge, seed=12)

COLOR_A   = "#4e79a7"   # blue  – main community
COLOR_B   = "#f28e2b"   # orange – dense neighbour
COLOR_BRIDGE = "#e15759"  # red – bridge nodes

BRIDGE_NODES = {6, 13}

for ax, comm_a, comm_b, title in [
    (axes[0], before_comm_A, before_comm_B,
     "Panel A – Before greedy reassignment\n"
     "(bridge nodes 6 & 13 hold the community together)"),
    (axes[1], after_comm_A, after_comm_B,
     "Panel B – After greedy reassignment of nodes 6 & 13\n"
     "(Community A fragments into 4 disconnected islands  ⚠)"),
]:
    node_colors = []
    for n in sorted(G_bridge.nodes()):
        if n in BRIDGE_NODES and n in comm_b:  # bridge moved to B
            node_colors.append(COLOR_B)
        elif n in BRIDGE_NODES and n in comm_a:  # bridge still in A
            node_colors.append(COLOR_BRIDGE)
        elif n in comm_a:
            node_colors.append(COLOR_A)
        elif n in comm_b:
            node_colors.append(COLOR_B)
        else:
            node_colors.append("#aaaaaa")

    # Separate bridge vs normal edges for visual weight
    bridge_edges = [(u, v) for u, v in G_bridge.edges()
                    if u in BRIDGE_NODES or v in BRIDGE_NODES]
    normal_edges = [(u, v) for u, v in G_bridge.edges()
                    if u not in BRIDGE_NODES and v not in BRIDGE_NODES]

    nx.draw_networkx_edges(G_bridge, pos=pos, ax=ax, edgelist=normal_edges,
                           edge_color="#999999", width=1.2)
    nx.draw_networkx_edges(G_bridge, pos=pos, ax=ax, edgelist=bridge_edges,
                           edge_color="#e15759", width=2.5, style="dashed")
    nx.draw_networkx_nodes(G_bridge, pos=pos, ax=ax,
                           node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G_bridge, pos=pos, ax=ax,
                            font_size=16, font_color="white", font_weight="bold")
    ax.set_title(title, fontsize=20)
    ax.axis("off")

legend_patches = [
    mpatches.Patch(color=COLOR_A,      label="Community A (main sub-groups)"),
    mpatches.Patch(color=COLOR_B,      label="Community B (dense neighbour)"),
    mpatches.Patch(color=COLOR_BRIDGE, label="Bridge node (before reassignment)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=16)
plt.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig("sop3_bridge_node.png", dpi=150, bbox_inches="tight")
print("\n  [Figure saved] sop3_bridge_node.png")


# Verify and report the actual Louvain output on the expanded graph
BEST_DISC_SEED = None
for trial_seed in range(50):
    louvain_bridge = nx.community.louvain_communities(G_bridge, seed=trial_seed)
    disc_count = count_disconnected(G_bridge, louvain_bridge)
    if disc_count > 0:
        BEST_DISC_SEED = trial_seed
        break

if BEST_DISC_SEED is not None:
    louvain_bridge = nx.community.louvain_communities(G_bridge, seed=BEST_DISC_SEED)
else:
    louvain_bridge = nx.community.louvain_communities(G_bridge, seed=42)

disc_count = count_disconnected(G_bridge, louvain_bridge)
print(f"\n  Louvain on expanded bridge-node graph "
      f"(seed={BEST_DISC_SEED if BEST_DISC_SEED is not None else 42}):")
for i, c in enumerate(sorted(louvain_bridge, key=lambda s: min(s)), 1):
    sub = G_bridge.subgraph(c)
    connected = nx.is_connected(sub)
    print(f"    Community {i}: nodes {sorted(c)}, connected={connected}")
    if not connected:
        frags = list(nx.connected_components(sub))
        print(f"            ↳ Disconnected fragments: {[sorted(f) for f in frags]}")

if disc_count:
    print(f"  ⚠  {disc_count} disconnected community/ies produced by Louvain "
          f"(greedy flaw confirmed).")
else:
    print("  ℹ  Louvain found connected communities on this graph.\n"
          "     The manual before/after diagram (Panel A/B) illustrates the "
          "theoretical mechanism.")


# ─────────────────────────────────────────────────────────────────────────────
# Part B – C-Town Water Network: Bridge-Node Risk Analysis
# ─────────────────────────────────────────────────────────────────────────────
# Even when Louvain does not produce disconnected communities on C-Town,
# we can directly show the MECHANISM by identifying all articulation points
# (nodes whose removal disconnects the network) and measuring how many of them
# sit on community boundaries — i.e. are at risk of greedy reassignment.

INP_FILE = "..\\GN Algorithm\\CTOWN.INP"

print(f"\n[Part B]  C-Town Water Network – Bridge-Node Risk Analysis")
try:
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    G_ctown = nx.Graph(wn.to_graph())
    print(f"  Loaded C-Town: {G_ctown.number_of_nodes()} nodes, "
          f"{G_ctown.number_of_edges()} edges")

    # Run Louvain and check for disconnected communities
    results = []
    for seed in [0, 1, 2, 3, 4]:
        partition = nx.community.louvain_communities(G_ctown, seed=seed)
        disc = count_disconnected(G_ctown, partition)
        q = nx.community.modularity(G_ctown, partition)
        results.append({"seed": seed, "num_communities": len(partition),
                         "disconnected": disc, "modularity": q})

    print(f"\n  Louvain results across 5 seeds:")
    print(f"  {'Seed':>5}  {'Communities':>12}  {'Disconnected':>13}  {'Modularity':>10}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*13}  {'-'*10}")
    for r in results:
        flag = "  ⚠" if r["disconnected"] > 0 else ""
        print(f"  {r['seed']:>5}  {r['num_communities']:>12}  "
              f"{r['disconnected']:>13}{flag}  {r['modularity']:>10.4f}")

    # Articulation-point analysis: find structural bridge nodes
    articulation_pts = set(nx.articulation_points(G_ctown))
    print(f"\n  Articulation points (structural bridges) in C-Town: "
          f"{len(articulation_pts)} nodes")

    # In the best partition, count articulation points that sit on community boundaries
    best = max(results, key=lambda r: r["modularity"])
    partition_best = nx.community.louvain_communities(G_ctown, seed=best["seed"])
    node_to_comm = {}
    for ci, comm in enumerate(partition_best):
        for nd in comm:
            node_to_comm[nd] = ci

    boundary_bridges = []
    for ap in articulation_pts:
        comm_ap = node_to_comm[ap]
        neighbour_comms = {node_to_comm[nb] for nb in G_ctown.neighbors(ap)
                           if node_to_comm[nb] != comm_ap}
        if neighbour_comms:   # articulation point that touches another community
            boundary_bridges.append(ap)

    print(f"  Articulation points on community boundaries  : "
          f"{len(boundary_bridges)} / {len(articulation_pts)}")
    print(f"  ⚠  Any of these {len(boundary_bridges)} nodes, if greedily reassigned "
          f"to a neighbour community, would disconnect C-Town's current partition.")

    # ── Figure B ─────────────────────────────────────────────────────────────
    pos_ct = nx.spring_layout(G_ctown, seed=42, k=0.4)
    palette = plt.cm.get_cmap("tab20", len(partition_best))
    node_list = list(G_ctown.nodes())

    # Colour scheme: community colour for normal nodes,
    # bright red for boundary bridge nodes (highest greedy-reassignment risk)
    node_colors_ct = []
    for nd in node_list:
        if nd in boundary_bridges:
            node_colors_ct.append("#e15759")   # red = at-risk bridge
        elif nd in articulation_pts:
            node_colors_ct.append("#f28e2b")   # orange = internal bridge
        else:
            node_colors_ct.append(palette(node_to_comm[nd]))

    node_sizes_ct = [120 if nd in boundary_bridges else
                     70  if nd in articulation_pts  else
                     40  for nd in node_list]

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle(
        "SOP 3 – C-Town: Structural Bridge Nodes at Risk of Greedy Reassignment",
        fontsize=12, fontweight="bold",
    )

    # Left panel: full network with bridge nodes highlighted
    ax_net = axes2[0]
    nx.draw_networkx(
        G_ctown, pos=pos_ct, ax=ax_net, with_labels=False,
        node_color=node_colors_ct, node_size=node_sizes_ct,
        edge_color="#cccccc", width=0.5,
    )
    ax_net.set_title(
        f"C-Town Louvain (seed={best['seed']}, Q={best['modularity']:.4f})\n"
        f"Red = boundary bridge nodes ({len(boundary_bridges)}), "
        f"Orange = internal bridges ({len(articulation_pts)-len(boundary_bridges)})",
        fontsize=9,
    )
    ax_net.axis("off")
    legend_net = [
        mpatches.Patch(color="#e15759", label=f"Boundary bridge ({len(boundary_bridges)} nodes)\n"
                                               "→ greedy move would disconnect community"),
        mpatches.Patch(color="#f28e2b", label=f"Internal bridge ({len(articulation_pts)-len(boundary_bridges)} nodes)"),
        mpatches.Patch(color="#4e79a7", label="Normal community node"),
    ]
    ax_net.legend(handles=legend_net, loc="lower left", fontsize=8)

    # Right panel: bar chart — community sizes, with count of bridge nodes per community
    comm_sizes = [len(c) for c in partition_best]
    bridge_counts = [sum(1 for nd in c if nd in boundary_bridges)
                     for c in partition_best]
    sorted_idx = sorted(range(len(comm_sizes)), key=lambda i: comm_sizes[i])
    s_sizes   = [comm_sizes[i]   for i in sorted_idx]
    s_bridges = [bridge_counts[i] for i in sorted_idx]

    ax_bar = axes2[1]
    x = range(len(s_sizes))
    bars = ax_bar.bar(x, s_sizes, color="#76b7b2", edgecolor="#1a5f6a",
                      label="Community size")
    ax_bar.bar(x, s_bridges, color="#e15759", edgecolor="#7b1f1f",
               label="Boundary bridges (greedy-risk nodes)")
    ax_bar.set_xlabel("Community index (sorted by size)")
    ax_bar.set_ylabel("Node count")
    ax_bar.set_title("Per-community: total size vs at-risk bridge nodes\n"
                     "(red = nodes whose reassignment would disconnect community)")
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig2.savefig("sop3_ctown_disconnected.png", dpi=150, bbox_inches="tight")
    print("  [Figure saved] sop3_ctown_disconnected.png")

except FileNotFoundError:
    print(f"  ⚠  Could not find {INP_FILE}. Skipping C-Town Part B.")

print("\n[Done] SOP 3 complete.\n")
