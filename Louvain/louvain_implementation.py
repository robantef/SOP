from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Set

import networkx as nx
import wntr


@dataclass
class _Status:
	node2com: Dict[object, int]
	degrees: Dict[int, float]
	gdegrees: Dict[object, float]
	internals: Dict[int, float]
	loops: Dict[object, float]
	total_weight: float


def _init_status(graph: nx.Graph, weight: str) -> _Status:
	node2com: Dict[object, int] = {}
	degrees: Dict[int, float] = {}
	gdegrees: Dict[object, float] = {}
	internals: Dict[int, float] = {}
	loops: Dict[object, float] = {}

	for idx, node in enumerate(graph.nodes()):
		node2com[node] = idx
		deg = float(graph.degree(node, weight=weight))
		loop_weight = float(graph.get_edge_data(node, node, default={}).get(weight, 0.0))
		degrees[idx] = deg
		gdegrees[node] = deg
		internals[idx] = loop_weight
		loops[node] = loop_weight

	return _Status(
		node2com=node2com,
		degrees=degrees,
		gdegrees=gdegrees,
		internals=internals,
		loops=loops,
		total_weight=float(graph.size(weight=weight)),
	)


def _renumber(node2com: Dict[object, int]) -> Dict[object, int]:
	mapper: Dict[int, int] = {}
	next_id = 0
	out: Dict[object, int] = {}
	for node, com in node2com.items():
		if com not in mapper:
			mapper[com] = next_id
			next_id += 1
		out[node] = mapper[com]
	return out


def _neighbor_communities(node: object, graph: nx.Graph, status: _Status, weight: str) -> Dict[int, float]:
	weights: Dict[int, float] = defaultdict(float)
	for nbr, edge_data in graph[node].items():
		if nbr == node:
			continue
		edge_weight = float(edge_data.get(weight, 1.0))
		weights[status.node2com[nbr]] += edge_weight
	return weights


def _remove(node: object, com: int, weight_to_com: float, status: _Status) -> None:
	status.degrees[com] = status.degrees.get(com, 0.0) - status.gdegrees[node]
	status.internals[com] = status.internals.get(com, 0.0) - weight_to_com - status.loops[node]
	status.node2com[node] = -1


def _insert(node: object, com: int, weight_to_com: float, status: _Status) -> None:
	status.node2com[node] = com
	status.degrees[com] = status.degrees.get(com, 0.0) + status.gdegrees[node]
	status.internals[com] = status.internals.get(com, 0.0) + weight_to_com + status.loops[node]


def _modularity(status: _Status) -> float:
	if status.total_weight <= 0:
		return 0.0

	result = 0.0
	communities = set(status.node2com.values())
	for com in communities:
		if com == -1:
			continue
		in_degree = status.internals.get(com, 0.0)
		degree = status.degrees.get(com, 0.0)
		result += (in_degree / status.total_weight) - (degree / (2.0 * status.total_weight)) ** 2
	return result


def _one_level(
	graph: nx.Graph,
	status: _Status,
	weight: str,
	resolution: float,
	min_gain: float,
	max_passes: int,
	rng: random.Random,
) -> bool:
	if status.total_weight <= 0:
		return False

	improvement = False
	current_modularity = _modularity(status)

	for _ in range(max_passes):
		moved = False
		nodes = list(graph.nodes())
		rng.shuffle(nodes)

		for node in nodes:
			current_com = status.node2com[node]
			neigh_com_weights = _neighbor_communities(node, graph, status, weight)
			weight_in_current = neigh_com_weights.get(current_com, 0.0)

			_remove(node, current_com, weight_in_current, status)

			best_com = current_com
			best_gain = 0.0
			node_degree = status.gdegrees[node]

			for com, wt in neigh_com_weights.items():
				gain = wt - resolution * status.degrees.get(com, 0.0) * node_degree / (2.0 * status.total_weight)
				if gain > best_gain:
					best_gain = gain
					best_com = com

			_insert(node, best_com, neigh_com_weights.get(best_com, 0.0), status)

			if best_com != current_com:
				moved = True
				improvement = True

		new_modularity = _modularity(status)
		if not moved or (new_modularity - current_modularity) < min_gain:
			break
		current_modularity = new_modularity

	return improvement


def _induced_graph(partition: Dict[object, int], graph: nx.Graph, weight: str) -> nx.Graph:
	induced = nx.Graph()
	for com in set(partition.values()):
		induced.add_node(com)

	for u, v, data in graph.edges(data=True):
		edge_weight = float(data.get(weight, 1.0))
		cu = partition[u]
		cv = partition[v]
		previous = float(induced.get_edge_data(cu, cv, default={}).get(weight, 0.0))
		induced.add_edge(cu, cv, **{weight: previous + edge_weight})

	return induced


def _partition_at_level(dendrogram: List[Dict[object, int]], level: int) -> Dict[object, int]:
	partition = dendrogram[0].copy()
	for idx in range(1, level + 1):
		next_level = dendrogram[idx]
		for node in partition:
			partition[node] = next_level[partition[node]]
	return partition


def louvain_communities(
	graph: nx.Graph,
	weight: str = "weight",
	resolution: float = 1.0,
	min_gain: float = 1e-7,
	max_passes: int = 100,
	seed: int = 42,
) -> List[Set[object]]:
	if graph.is_directed():
		raise ValueError("This implementation expects an undirected graph.")

	if graph.number_of_nodes() == 0:
		return []

	rng = random.Random(seed)
	current_graph = graph.copy()
	dendrogram: List[Dict[object, int]] = []

	while True:
		status = _init_status(current_graph, weight)
		improved = _one_level(
			graph=current_graph,
			status=status,
			weight=weight,
			resolution=resolution,
			min_gain=min_gain,
			max_passes=max_passes,
			rng=rng,
		)

		partition = _renumber(status.node2com)
		dendrogram.append(partition)

		if not improved or len(set(partition.values())) == current_graph.number_of_nodes():
			break

		current_graph = _induced_graph(partition, current_graph, weight)

	final_partition = _partition_at_level(dendrogram, len(dendrogram) - 1)
	grouped: Dict[int, Set[object]] = defaultdict(set)
	for node, com in final_partition.items():
		grouped[com].add(node)

	communities = sorted(grouped.values(), key=len, reverse=True)
	return communities


def main() -> None:
	inp_file = "..\\GN Algorithm\\CTOWN.INP"
	wn = wntr.network.WaterNetworkModel(inp_file)
	graph = nx.Graph(wn.to_graph())

	print(f"Loaded C-Town: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

	communities = louvain_communities(graph)

	print(f"Found {len(communities)} Louvain communities:")
	for idx, community in enumerate(communities, 1):
		print(f"Community {idx}: size={len(community)}")


if __name__ == "__main__":
	main()
