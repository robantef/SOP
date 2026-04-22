from collections import defaultdict
import random

import networkx as nx


def build_sample_graph() -> nx.Graph:
	graph = nx.Graph()

	nodes = ["1", "2", "3", "4", "5"]
	edges = [
		("1", "2", 1),
		("1", "3", 1),
		("2", "3", 1),
		("2", "4", 1),
		("3", "4", 1),
		("3", "5", 1),
		("4", "2", 1),
		("4", "5", 1),
	]

	graph.add_nodes_from(nodes)
	graph.add_edges_from(edges)
	return graph


def print_partition(partition, title: str) -> None:
	print(title)
	for i, community in enumerate(partition, 1):
		sorted_nodes = sorted(community)
		print(f"  Community {i}: {sorted_nodes}")


def _neighbor_weights(nbrs, node2com):
	weights = defaultdict(float)
	for nbr, wt in nbrs.items():
		weights[node2com[nbr]] += wt
	return weights


def _gen_graph(graph: nx.Graph, partition):
	induced = graph.__class__()
	node2com = {}
	for i, part in enumerate(partition):
		nodes = set()
		for node in part:
			node2com[node] = i
			nodes.update(graph.nodes[node].get("nodes", {node}))
		induced.add_node(i, nodes=nodes)

	for node1, node2, wt in graph.edges(data=True):
		edge_weight = wt["weight"]
		com1 = node2com[node1]
		com2 = node2com[node2]
		prev_weight = induced.get_edge_data(com1, com2, {"weight": 0})["weight"]
		induced.add_edge(com1, com2, weight=edge_weight + prev_weight)
	return induced


def _one_level_trace(graph: nx.Graph, m: float, partition, resolution: float, rng: random.Random, node_order=None):
	node2com = {u: i for i, u in enumerate(graph.nodes())}
	inner_partition = [{u} for u in graph.nodes()]

	def _display_com(com_id: int) -> int:
		# Internal community ids are 0-based; show 1-based labels in trace output.
		return com_id + 1

	degrees = dict(graph.degree(weight="weight"))
	stot = list(degrees.values())
	nbrs = {u: {v: data["weight"] for v, data in graph[u].items() if v != u} for u in graph}

	if node_order is None:
		rand_nodes = list(graph.nodes)
		rng.shuffle(rand_nodes)
	else:
		rand_nodes = list(node_order)
	print(f"  Node visiting order: {rand_nodes}")

	nb_moves = 1
	improvement = False
	pass_no = 0

	while nb_moves > 0:
		pass_no += 1
		print(f"\n  Pass {pass_no}")
		nb_moves = 0

		for u in rand_nodes:
			best_mod = 0.0
			best_com = node2com[u]
			current_com = node2com[u]
			weights2com = _neighbor_weights(nbrs[u], node2com)
			degree = degrees[u]

			stot[current_com] -= degree
			remove_cost = -weights2com[current_com] / m + resolution * (stot[current_com] * degree) / (2 * m * m)

			display_weights = {_display_com(com): wt for com, wt in sorted(weights2com.items())}
			print(f"    Node {u}: current community {_display_com(current_com)}")
			print(f"      Neighbor-community edge weights: {display_weights}")
			print(f"      remove_cost = {remove_cost:.6f}")

			for nbr_com, wt in sorted(weights2com.items()):
				gain = remove_cost + wt / m - resolution * (stot[nbr_com] * degree) / (2 * m * m)
				print(f"      Try community {_display_com(nbr_com)}: gain = {gain:.6f}")
				if gain > best_mod:
					best_mod = gain
					best_com = nbr_com

			stot[best_com] += degree

			if best_com != current_com:
				print(
					f"      Move node {u}: "
					f"{_display_com(current_com)} -> {_display_com(best_com)} "
					f"(best_gain={best_mod:.6f})"
				)
				com_nodes = graph.nodes[u].get("nodes", {u})
				partition[current_com].difference_update(com_nodes)
				inner_partition[current_com].remove(u)
				partition[best_com].update(com_nodes)
				inner_partition[best_com].add(u)
				node2com[u] = best_com
				improvement = True
				nb_moves += 1
			else:
				print(f"      Keep node {u} in community {_display_com(current_com)}")

		current_partition = [s.copy() for s in partition if s]
		pass_modularity = nx.community.modularity(graph, [s.copy() for s in inner_partition if s], resolution=resolution, weight="weight")
		print_partition(current_partition, f"  Partition after pass {pass_no}:")
		print(f"  Modularity after pass {pass_no}: {pass_modularity:.6f}")
		print(f"  Moves this pass: {nb_moves}")

	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))
	return partition, inner_partition, improvement


def traced_louvain_partitions(graph: nx.Graph, resolution: float = 1.0, threshold: float = 1e-7, seed: int = 42):
	if graph.is_directed():
		raise ValueError("This traced script supports undirected graphs only.")

	rng = random.Random(seed)
	partition = [{u} for u in graph.nodes()]

	if nx.is_empty(graph):
		yield partition
		return

	mod = nx.community.modularity(graph, partition, resolution=resolution, weight="weight")
	working_graph = graph.__class__()
	working_graph.add_nodes_from(graph)
	working_graph.add_edges_from(graph.edges(data="weight", default=1))
	m = working_graph.size(weight="weight")

	level = 1
	while True:
		print(f"\nStep 3.{level}: Optimize Louvain level {level}")
		print(f"  Working graph: {working_graph.number_of_nodes()} nodes, {working_graph.number_of_edges()} edges")
		forced_order = sorted(working_graph.nodes()) if level == 1 else None

		partition, inner_partition, improvement = _one_level_trace(
			working_graph,
			m,
			partition,
			resolution,
			rng,
			forced_order,
		)

		yield [s.copy() for s in partition]

		new_mod = nx.community.modularity(
			working_graph,
			inner_partition,
			resolution=resolution,
			weight="weight",
		)
		gain = new_mod - mod
		print(f"  Level {level} modularity on current graph: {new_mod:.6f}")
		print(f"  Modularity gain from previous level: {gain:.6f}")

		if (gain <= threshold) or (not improvement):
			print("  Stop condition reached (low modularity gain or no improvement).")
			return

		mod = new_mod
		working_graph = _gen_graph(working_graph, inner_partition)
		print(
			f"  Build induced graph for next level: "
			f"{working_graph.number_of_nodes()} nodes, {working_graph.number_of_edges()} edges"
		)
		level += 1


def main() -> None:
	print("Step 1: Build the sample graph")
	graph = build_sample_graph()

	print(f"  Nodes: {sorted(graph.nodes())}")
	print("  Edges (u, v, weight):")
	for u, v, w in sorted(graph.edges(data="weight")):
		print(f"    ({u}, {v}, {w})")

	print("\nStep 2: Initial partition (each node is its own community)")
	initial_partition = [{node} for node in sorted(graph.nodes())]
	print_partition(initial_partition, "  Initial communities:")
	initial_modularity = nx.community.modularity(graph, initial_partition, weight="weight")
	print(f"  Initial modularity: {initial_modularity:.6f}")

	print("\nStep 3: Detailed Louvain optimization trace")
	traced_partitions = list(traced_louvain_partitions(graph, resolution=1.0, threshold=1e-7, seed=42))

	if not traced_partitions:
		print("  No Louvain levels returned.")
		return

	print("\nStep 4: Final traced result")
	final_partition = traced_partitions[-1]
	print_partition(final_partition, "  Final communities from detailed trace:")

	node_to_community = {}
	for idx, community in enumerate(final_partition, 1):
		for node in community:
			node_to_community[node] = idx

	print("  Node -> community assignment:")
	for node in sorted(node_to_community):
		print(f"    {node} -> Community {node_to_community[node]}")

	nx_partition = nx.community.louvain_communities(graph, weight="weight", seed=42, resolution=1.0)
	print_partition(nx_partition, "\nStep 5: NetworkX built-in Louvain result:")


if __name__ == "__main__":
	main()
