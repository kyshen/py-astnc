from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set

import networkx as nx

from astnc.internal.network import TensorNetwork


@dataclass
class PartitionNode:
    node_ids: FrozenSet[int]
    children: Optional[List["PartitionNode"]]
    boundary_labels: List[int]
    open_labels: List[int]
    cut_labels: List[int]

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def node_key(self) -> tuple[int, ...]:
        return tuple(sorted(self.node_ids))

    @property
    def subtree_size(self) -> int:
        return len(self.node_ids)


def _subtree_boundary_labels(tn: TensorNetwork, node_ids: Set[int]) -> List[int]:
    output: List[int] = []
    for label, attached in tn.label_to_nodes.items():
        inside = sum(1 for node_id in attached if node_id in node_ids)
        if inside == 1 and len(attached) == 2:
            output.append(label)
    return sorted(output)


def _subtree_open_labels(tn: TensorNetwork, node_ids: Set[int]) -> List[int]:
    labels: List[int] = []
    for node_id in sorted(node_ids):
        labels.extend(tn.node_map[node_id].open_labels)
    order = {label: idx for idx, label in enumerate(tn.open_label_order)}
    return sorted(labels, key=lambda label: order[label])


def _cut_labels_between(tn: TensorNetwork, left: Set[int], right: Set[int]) -> List[int]:
    output: List[int] = []
    for label, attached in tn.label_to_nodes.items():
        if len(attached) == 2:
            a, b = attached
            if (a in left and b in right) or (a in right and b in left):
                output.append(label)
    return sorted(output)


def _recursive_build(tn: TensorNetwork, node_ids: Set[int]) -> PartitionNode:
    boundary_labels = _subtree_boundary_labels(tn, node_ids)
    open_labels = _subtree_open_labels(tn, node_ids)
    if len(node_ids) == 1:
        return PartitionNode(frozenset(node_ids), None, boundary_labels, open_labels, [])

    subgraph = tn.interaction_graph().subgraph(node_ids).copy()
    if subgraph.number_of_edges() == 0:
        items = list(node_ids)
        left_nodes, right_nodes = {items[0]}, set(items[1:])
    else:
        _, parts = nx.stoer_wagner(subgraph, weight="weight")
        left_nodes = set(parts[0])
        right_nodes = set(parts[1])
        if not left_nodes or not right_nodes:
            items = list(node_ids)
            left_nodes, right_nodes = {items[0]}, set(items[1:])

    cut_labels = _cut_labels_between(tn, left_nodes, right_nodes)
    left = _recursive_build(tn, left_nodes)
    right = _recursive_build(tn, right_nodes)
    return PartitionNode(frozenset(node_ids), [left, right], boundary_labels, open_labels, cut_labels)


def build_partition_tree(tn: TensorNetwork) -> PartitionNode:
    return _recursive_build(tn, set(node.node_id for node in tn.nodes))

