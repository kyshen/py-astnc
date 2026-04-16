from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import opt_einsum as oe


@dataclass
class TensorNode:
    node_id: int
    tensor: np.ndarray
    labels: List[int]
    open_labels: List[int]
    internal_labels: List[int]


class TensorNetwork:
    def __init__(
        self,
        nodes: Sequence[TensorNode],
        label_dims: Dict[int, int],
        open_label_order: List[int],
        label_to_nodes: Dict[int, List[int]],
    ) -> None:
        self.nodes = list(nodes)
        self.label_dims = dict(label_dims)
        self.open_label_order = list(open_label_order)
        self.label_to_nodes = {int(k): list(v) for k, v in label_to_nodes.items()}
        self.node_map = {n.node_id: n for n in self.nodes}

    @property
    def num_open(self) -> int:
        return len(self.open_label_order)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return tuple(int(self.label_dims[label]) for label in self.open_label_order)

    def interaction_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for node in self.nodes:
            graph.add_node(node.node_id)
        for label, attached in self.label_to_nodes.items():
            if len(attached) == 2:
                u, v = attached
                weight = float(self.label_dims[label])
                if graph.has_edge(u, v):
                    graph[u][v]["weight"] += weight
                    graph[u][v]["labels"].append(label)
                else:
                    graph.add_edge(u, v, weight=weight, labels=[label])
        return graph

    def _prepared_operands(
        self,
        node_ids: Iterable[int],
        slice_map: Dict[int, Sequence[int]] | None = None,
    ) -> list[tuple[np.ndarray, list[int]]]:
        operands = []
        slice_map = slice_map or {}
        for node_id in set(node_ids):
            node = self.node_map[node_id]
            tensor = node.tensor
            labels = list(node.labels)
            for axis, label in enumerate(labels):
                if label in slice_map:
                    index = np.asarray(slice_map[label], dtype=int)
                    tensor = np.take(tensor, index, axis=axis)
            operands.append((tensor, labels))
        return operands

    def contract_subnetwork(
        self,
        node_ids: Iterable[int],
        output_labels: Sequence[int],
        slice_map: Dict[int, Sequence[int]] | None = None,
        optimize: str = "optimal",
    ) -> np.ndarray:
        operands = self._prepared_operands(node_ids, slice_map=slice_map)
        args: list[object] = []
        for tensor, labels in operands:
            args.append(tensor)
            args.append(labels)
        args.append(list(output_labels))
        return oe.contract(*args, optimize=optimize)

    def contract_full(
        self,
        slice_map: Dict[int, Sequence[int]] | None = None,
        optimize: str = "optimal",
    ) -> np.ndarray:
        return self.contract_subnetwork(
            [node.node_id for node in self.nodes],
            self.open_label_order,
            slice_map=slice_map,
            optimize=optimize,
        )

