from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping

from astnc.internal.network import TensorNetwork


@dataclass
class OutputBlock:
    block_id: int
    slice_map: Dict[int, List[int]]


def make_blocks(tn: TensorNetwork, block_spec: Mapping[int, int] | None = None) -> List[OutputBlock]:
    resolved = {int(axis): int(size) for axis, size in dict(block_spec or {}).items()}
    if not resolved:
        return [OutputBlock(block_id=0, slice_map={})]

    if min(resolved) < 0 or max(resolved) >= len(tn.open_label_order):
        raise ValueError("`block_spec` keys must be valid output-axis indices.")

    axes = sorted(resolved)
    labels = [tn.open_label_order[axis] for axis in axes]
    per_label_chunks: List[List[List[int]]] = []
    for axis, label in zip(axes, labels):
        chunk_size = int(resolved[axis])
        if chunk_size <= 0:
            raise ValueError("`block_spec` chunk sizes must be positive integers.")
        dim = int(tn.label_dims[label])
        chunks = [list(range(i, min(i + chunk_size, dim))) for i in range(0, dim, chunk_size)]
        per_label_chunks.append(chunks)

    output: List[OutputBlock] = []
    for block_id, combo in enumerate(itertools.product(*per_label_chunks)):
        slice_map = {label: indices for label, indices in zip(labels, combo)}
        output.append(OutputBlock(block_id=block_id, slice_map=slice_map))
    return output
