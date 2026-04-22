"""Run Louvain community detection with python-igraph.

This script loads the C-Town EPANET network and applies igraph's
multilevel community detection (Louvain algorithm).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import igraph as ig
import networkx as nx
import wntr


def load_ctown_as_igraph(inp_path: str) -> ig.Graph:
	"""Load an EPANET INP file and convert it to an undirected igraph graph."""
	wn = wntr.network.WaterNetworkModel(inp_path)

	# Convert to a simple undirected NetworkX graph first, then map to igraph.
	nx_graph = nx.Graph(wn.to_graph())

	node_names = list(nx_graph.nodes())
	node_to_idx = {name: idx for idx, name in enumerate(node_names)}
	edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]

	graph = ig.Graph(n=len(node_names), edges=edges, directed=False)
	graph.vs["name"] = node_names
	return graph


def run_louvain(graph: ig.Graph, resolution: float = 1.0, seed: int | None = None) -> ig.VertexClustering:
	"""Run Louvain (community_multilevel) on an igraph graph."""
	if seed is not None:
		ig.set_random_number_generator(random.Random(seed))

	return graph.community_multilevel(resolution=resolution)


def print_partition(graph: ig.Graph, clustering: ig.VertexClustering, preview: int = 8) -> None:
	"""Print partition summary and per-community node previews."""
	communities = [list(comm) for comm in clustering]
	communities.sort(key=len, reverse=True)

	print(f"Found {len(communities)} communities")
	print(f"Modularity: {clustering.modularity:.6f}")
	print()

	for idx, community in enumerate(communities, start=1):
		names = [graph.vs[v]["name"] for v in community]
		sample = ", ".join(names[:preview])
		suffix = "" if len(names) <= preview else ", ..."
		print(f"Community {idx}: size={len(names)}")
		print(f"  Nodes: {sample}{suffix}")


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""
	default_inp = Path(__file__).resolve().parents[2] / "GN Algorithm" / "CTOWN.INP"

	parser = argparse.ArgumentParser(description="Louvain community detection using python-igraph")
	parser.add_argument(
		"--inp",
		type=str,
		default=str(default_inp),
		help="Path to EPANET .INP file (default: GN Algorithm/CTOWN.INP)",
	)
	parser.add_argument(
		"--resolution",
		type=float,
		default=1.0,
		help="Resolution parameter (lower -> larger communities, higher -> smaller communities)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional random seed for reproducibility",
	)
	parser.add_argument(
		"--preview",
		type=int,
		default=8,
		help="How many node names to print per community",
	)
	return parser.parse_args()


def main() -> None:
	"""Entry point for running Louvain with igraph."""
	args = parse_args()

	graph = load_ctown_as_igraph(args.inp)
	print(f"Loaded graph: {graph.vcount()} nodes, {graph.ecount()} edges")

	clustering = run_louvain(graph, resolution=args.resolution, seed=args.seed)
	print_partition(graph, clustering, preview=max(1, args.preview))


if __name__ == "__main__":
	main()
