from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from astnc.internal.network import TensorNetwork, TensorNode


def _ensure_connected_random(num_nodes: int, edge_prob: float, rng: np.random.Generator) -> nx.Graph:
    while True:
        graph = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=int(rng.integers(0, 1_000_000)))
        if nx.is_connected(graph):
            return graph


def _build_tensor_network(
    graph: nx.Graph,
    *,
    phys_dim: int,
    bond_dim: int,
    open_legs_per_node: int,
    seed: int,
) -> TensorNetwork:
    rng = np.random.default_rng(seed)
    next_label = 0
    label_dims: Dict[int, int] = {}
    label_to_nodes: Dict[int, List[int]] = {}
    open_label_order: List[int] = []
    internal_edge_to_label: Dict[Tuple[int, int], int] = {}

    for u, v in graph.edges():
        key = tuple(sorted((int(u), int(v))))
        label = next_label
        next_label += 1
        internal_edge_to_label[key] = label
        label_dims[label] = bond_dim
        label_to_nodes[label] = [int(u), int(v)]

    nodes: List[TensorNode] = []
    for node_id in sorted(graph.nodes()):
        labels: List[int] = []
        open_labels: List[int] = []
        internal_labels: List[int] = []
        for _ in range(open_legs_per_node):
            label = next_label
            next_label += 1
            labels.append(label)
            open_labels.append(label)
            open_label_order.append(label)
            label_dims[label] = phys_dim
            label_to_nodes[label] = [int(node_id)]
        for neighbor in sorted(graph.neighbors(node_id)):
            label = internal_edge_to_label[tuple(sorted((int(node_id), int(neighbor))))]
            labels.append(label)
            internal_labels.append(label)
        shape = tuple(label_dims[label] for label in labels)
        scale = np.sqrt(max(1, np.prod(shape)))
        tensor = rng.standard_normal(shape).astype(np.float64) / scale
        nodes.append(
            TensorNode(
                node_id=int(node_id),
                tensor=tensor,
                labels=labels,
                open_labels=open_labels,
                internal_labels=internal_labels,
            )
        )
    return TensorNetwork(nodes=nodes, label_dims=label_dims, open_label_order=open_label_order, label_to_nodes=label_to_nodes)


def random_connected(
    *,
    num_nodes: int,
    phys_dim: int,
    bond_dim: int,
    open_legs_per_node: int = 1,
    edge_prob: float = 0.35,
    seed: int = 0,
) -> TensorNetwork:
    rng = np.random.default_rng(seed)
    graph = _ensure_connected_random(num_nodes, edge_prob, rng)
    return _build_tensor_network(
        graph,
        phys_dim=phys_dim,
        bond_dim=bond_dim,
        open_legs_per_node=open_legs_per_node,
        seed=seed,
    )


def ring(
    *,
    num_nodes: int,
    phys_dim: int,
    bond_dim: int,
    open_legs_per_node: int = 1,
    seed: int = 0,
) -> TensorNetwork:
    return _build_tensor_network(
        nx.cycle_graph(num_nodes),
        phys_dim=phys_dim,
        bond_dim=bond_dim,
        open_legs_per_node=open_legs_per_node,
        seed=seed,
    )


def tree(
    *,
    num_nodes: int,
    phys_dim: int,
    bond_dim: int,
    open_legs_per_node: int = 1,
    seed: int = 0,
) -> TensorNetwork:
    rng = np.random.default_rng(seed)
    graph = nx.random_labeled_tree(num_nodes, seed=int(rng.integers(0, 1_000_000)))
    return _build_tensor_network(
        graph,
        phys_dim=phys_dim,
        bond_dim=bond_dim,
        open_legs_per_node=open_legs_per_node,
        seed=seed,
    )


def grid2d(
    *,
    rows: int,
    cols: int,
    phys_dim: int,
    bond_dim: int,
    open_legs_per_node: int = 1,
    seed: int = 0,
) -> TensorNetwork:
    base = nx.grid_2d_graph(rows, cols)
    mapping = {node: idx for idx, node in enumerate(base.nodes())}
    graph = nx.relabel_nodes(base, mapping)
    return _build_tensor_network(
        graph,
        phys_dim=phys_dim,
        bond_dim=bond_dim,
        open_legs_per_node=open_legs_per_node,
        seed=seed,
    )

