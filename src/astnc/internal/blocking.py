from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List

from astnc.internal.network import TensorNetwork


@dataclass
class OutputBlock:
    block_id: int
    slice_map: Dict[int, List[int]]


def make_blocks(tn: TensorNetwork, block_label_count: int, chunk_size: int) -> List[OutputBlock]:
    labels = tn.open_label_order[: int(block_label_count)]
    if len(labels) == 0:
        return [OutputBlock(block_id=0, slice_map={})]

    per_label_chunks: List[List[List[int]]] = []
    for label in labels:
        dim = int(tn.label_dims[label])
        chunks = [list(range(i, min(i + int(chunk_size), dim))) for i in range(0, dim, int(chunk_size))]
        per_label_chunks.append(chunks)

    output: List[OutputBlock] = []
    for block_id, combo in enumerate(itertools.product(*per_label_chunks)):
        slice_map = {label: indices for label, indices in zip(labels, combo)}
        output.append(OutputBlock(block_id=block_id, slice_map=slice_map))
    return output

