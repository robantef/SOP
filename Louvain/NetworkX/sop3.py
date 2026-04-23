"""
SOP 3 (Part A only): Bridge-node reassignment can disconnect a community.

This script is for bridge-node demonstration:
1) Build a small graph with one bridge node.
2) Show that the source community is connected before reassignment.
3) Reassign the bridge node to a neighboring community.
4) Show that the source community becomes disconnected.
5) Save a simple before/after figure.
"""

import matplotlib
import networkx as nx

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_bridge_demo_graph() -> nx.Graph:
	"""Create a small graph where one node acts as a structural bridge."""
	g = nx.Graph()

	# Main community has two dense subgroups connected only through node X.
	g.add_edges_from([
		("A", "B"),
		("B", "C"),
		("A", "C"),  # left triangle
		("D", "E"),
		("E", "F"),
		("D", "F"),  # right triangle
		("C", "X"),
		("D", "X"),  # X is the sole connector between left/right
	])

	# Neighbor community (dense) that can attract X during greedy moves.
	g.add_edges_from([
		("P", "Q"),
		("Q", "R"),
		("P", "R"),  # triangle
		("X", "P"),
		("X", "Q"),  # extra pull toward neighbor community
	])

	return g


def is_connected_subgraph(g: nx.Graph, nodes: set[str]) -> bool:
	"""Connectivity test for one community."""
	sub = g.subgraph(nodes)
	if sub.number_of_nodes() == 0:
		return True
	return nx.is_connected(sub)


def save_before_after_figure(
	g: nx.Graph,
	before_main: set[str],
	before_neighbor: set[str],
	after_main: set[str],
	after_neighbor: set[str],
	out_file: str = "sop3.png",
) -> None:
	"""Plot the partition before/after moving bridge node X."""
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	fig.suptitle("SOP 3 Part A: Bridge-node reassignment disconnects a community")

	pos = nx.spring_layout(g, seed=10)
	bridge_node = "X"

	def draw_panel(ax, main_comm: set[str], neighbor_comm: set[str], title: str) -> None:
		colors = []
		for n in g.nodes():
			if n == bridge_node and n in main_comm:
				colors.append("#e15759")  # red: bridge in main community
			elif n in main_comm:
				colors.append("#4e79a7")  # blue: main community
			elif n in neighbor_comm:
				colors.append("#f28e2b")  # orange: neighbor community
			else:
				colors.append("#999999")

		nx.draw_networkx_edges(g, pos=pos, ax=ax, edge_color="#888888", width=1.5)
		nx.draw_networkx_nodes(g, pos=pos, ax=ax, node_color=colors, node_size=700)
		nx.draw_networkx_labels(g, pos=pos, ax=ax, font_color="white", font_weight="bold")
		ax.set_title(title)
		ax.axis("off")

	draw_panel(
		axes[0],
		before_main,
		before_neighbor,
		"Before reassignment\nX belongs to main community",
	)
	draw_panel(
		axes[1],
		after_main,
		after_neighbor,
		"After reassignment\nX moved to neighbor community",
	)

	plt.tight_layout()
	fig.savefig(out_file, dpi=150, bbox_inches="tight")


def main() -> None:
	print("=" * 62)
	print("SOP 3 Part A - Bridge Node Reassignment Demonstration")
	print("=" * 62)

	g = build_bridge_demo_graph()
	print(f"Nodes: {g.number_of_nodes()} | Edges: {g.number_of_edges()}")

	# Before: X is part of the main community.
	before_main = {"A", "B", "C", "D", "E", "F", "X"}
	before_neighbor = {"P", "Q", "R"}

	# After greedy move: X is reassigned to the neighbor community.
	after_main = {"A", "B", "C", "D", "E", "F"}
	after_neighbor = {"P", "Q", "R", "X"}

	before_connected = is_connected_subgraph(g, before_main)
	after_connected = is_connected_subgraph(g, after_main)

	print("\nConnectivity check for main community:")
	print(f"  Before reassignment: connected = {before_connected}")
	print(f"  After reassignment : connected = {after_connected}")

	if not after_connected:
		comps = [sorted(c) for c in nx.connected_components(g.subgraph(after_main))]
		print(f"  Main community fragments: {comps}")

	save_before_after_figure(
		g,
		before_main,
		before_neighbor,
		after_main,
		after_neighbor,
	)
	print("\nFigure saved: sop3.png")


if __name__ == "__main__":
	main()
