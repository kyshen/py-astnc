from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from astnc.internal.blocking import OutputBlock
from astnc.internal.cache import SeparatorStateCache, StateCacheKey, make_local_block_key
from astnc.internal.network import TensorNetwork
from astnc.internal.partition import PartitionNode, build_partition_tree
from astnc.internal.state import (
    MergeInfo,
    SeparatorState,
    adaptive_state_from_tensor,
    merge_states,
)


@dataclass
class ContractionResult:
    dense: np.ndarray
    contract_time_sec: float
    emit_time_sec: float
    meta: Dict[str, Any]


@dataclass
class ASTNCRuntimeStats:
    leaf_states_built: int = 0
    internal_states_built: int = 0
    num_compressed_merges: int = 0
    num_implicit_merge_sketches: int = 0
    num_explicit_merge_compressions: int = 0
    num_exact_leaves: int = 0
    num_compressed_leaves: int = 0

    def observe(self, *, is_leaf: bool) -> None:
        if is_leaf:
            self.leaf_states_built += 1
        else:
            self.internal_states_built += 1

    def observe_leaf_choice(self, *, compressed: bool) -> None:
        if compressed:
            self.num_compressed_leaves += 1
        else:
            self.num_exact_leaves += 1

    def observe_merge(self, merge_info: MergeInfo) -> None:
        if merge_info.compressed:
            self.num_compressed_merges += 1
            if merge_info.path == "implicit_randomized":
                self.num_implicit_merge_sketches += 1
            else:
                self.num_explicit_merge_compressions += 1

    def summary(self) -> Dict[str, int | float]:
        return {
            "leaf_states_built": int(self.leaf_states_built),
            "internal_states_built": int(self.internal_states_built),
            "num_compressed_merges": int(self.num_compressed_merges),
            "num_implicit_merge_sketches": int(self.num_implicit_merge_sketches),
            "num_explicit_merge_compressions": int(self.num_explicit_merge_compressions),
            "num_exact_leaves": int(self.num_exact_leaves),
            "num_compressed_leaves": int(self.num_compressed_leaves),
        }


@dataclass
class ASTNCTreeTrace:
    tree: Dict[str, Any]
    blocks: List[Dict[str, Any]]
    _node_map: Dict[tuple[int, ...], Dict[str, Any]]
    _state_meta: Dict[StateCacheKey, Dict[str, Any]]

    @classmethod
    def build(
        cls,
        part: PartitionNode,
        depth_info: Dict[tuple[int, ...], tuple[int, int, int]],
    ) -> "ASTNCTreeTrace":
        node_map: Dict[tuple[int, ...], Dict[str, Any]] = {}

        def build_node(current: PartitionNode) -> Dict[str, Any]:
            depth, _, subtree_size = depth_info.get(current.node_key, (0, len(current.open_labels), current.subtree_size))
            node = {
                "node_key": list(current.node_key),
                "node_ids": sorted(int(node_id) for node_id in current.node_ids),
                "depth": int(depth),
                "subtree_size": int(subtree_size),
                "is_leaf": bool(current.is_leaf),
                "open_labels": [int(label) for label in current.open_labels],
                "boundary_labels": [int(label) for label in current.boundary_labels],
                "cut_labels": [int(label) for label in current.cut_labels],
                "per_block": [],
                "children": [],
            }
            node_map[current.node_key] = node
            if current.children:
                node["children"] = [build_node(child) for child in current.children]
            return node

        tree = build_node(part)
        return cls(tree=tree, blocks=[], _node_map=node_map, _state_meta={})

    def register_block(self, block_index: int, block: OutputBlock) -> None:
        self.blocks.append(
            {
                "block_index": int(block_index),
                "slice_map": {
                    int(label): [int(idx) for idx in indices]
                    for label, indices in sorted(block.slice_map.items())
                },
            }
        )

    def _entry_base(
        self,
        *,
        part: PartitionNode,
        block_index: int,
        slice_map,
        state: SeparatorState,
        actual_tol: float,
        source: str,
        path: str,
    ) -> Dict[str, Any]:
        return {
            "block_index": int(block_index),
            "local_slice_map": {
                int(label): [int(idx) for idx in slice_map[label]]
                for label in part.open_labels
                if label in slice_map
            },
            "source": str(source),
            "path": str(path),
            "rank": int(state.rank),
            "actual_tol": float(actual_tol),
        }

    def record_leaf(
        self,
        *,
        state_key: StateCacheKey,
        part: PartitionNode,
        block_index: int,
        slice_map,
        state: SeparatorState,
        actual_tol: float,
        full_rank: int,
        residual_ratio: float | None,
        source: str,
    ) -> None:
        entry = self._entry_base(
            part=part,
            block_index=block_index,
            slice_map=slice_map,
            state=state,
            actual_tol=actual_tol,
            source=source,
            path="leaf",
        )
        entry["full_rank"] = int(full_rank)
        entry["compressed"] = bool(state.rank < full_rank)
        entry["residual_ratio"] = None if residual_ratio is None else float(residual_ratio)
        self._node_map[part.node_key]["per_block"].append(entry)
        if source == "computed":
            self._state_meta[state_key] = dict(entry)

    def record_merge(
        self,
        *,
        state_key: StateCacheKey,
        part: PartitionNode,
        block_index: int,
        slice_map,
        state: SeparatorState,
        actual_tol: float,
        merge_info: MergeInfo | None,
        source: str,
    ) -> None:
        path = "cache" if merge_info is None else str(merge_info.path)
        entry = self._entry_base(
            part=part,
            block_index=block_index,
            slice_map=slice_map,
            state=state,
            actual_tol=actual_tol,
            source=source,
            path=path,
        )
        entry["full_rank"] = None if merge_info is None else int(merge_info.full_rank)
        entry["compressed"] = None if merge_info is None else bool(merge_info.compressed)
        entry["residual_ratio"] = None if merge_info is None else float(merge_info.residual_ratio)
        self._node_map[part.node_key]["per_block"].append(entry)
        if source == "computed":
            self._state_meta[state_key] = dict(entry)

    def record_cache_hit(
        self,
        *,
        state_key: StateCacheKey,
        part: PartitionNode,
        block_index: int,
        slice_map,
        state: SeparatorState,
        actual_tol: float,
    ) -> None:
        cached_meta = self._state_meta.get(state_key)
        if cached_meta is None:
            if part.is_leaf:
                open_flat = int(np.prod(state.open_dims)) if state.open_dims else 1
                boundary_flat = int(np.prod(state.boundary_dims)) if state.boundary_dims else 1
                self.record_leaf(
                    state_key=state_key,
                    part=part,
                    block_index=block_index,
                    slice_map=slice_map,
                    state=state,
                    actual_tol=actual_tol,
                    full_rank=min(open_flat, boundary_flat),
                    residual_ratio=None,
                    source="cache_hit",
                )
                return
            self.record_merge(
                state_key=state_key,
                part=part,
                block_index=block_index,
                slice_map=slice_map,
                state=state,
                actual_tol=actual_tol,
                merge_info=None,
                source="cache_hit",
            )
            return

        entry = dict(cached_meta)
        entry["block_index"] = int(block_index)
        entry["local_slice_map"] = {
            int(label): [int(idx) for idx in slice_map[label]]
            for label in part.open_labels
            if label in slice_map
        }
        entry["source"] = "cache_hit"
        entry["actual_tol"] = float(actual_tol)
        self._node_map[part.node_key]["per_block"].append(entry)


def exact_contraction(tn: TensorNetwork, optimize: str = "optimal") -> ContractionResult:
    import time

    t0 = time.perf_counter()
    dense = tn.contract_full(optimize=optimize)
    t_contract = time.perf_counter() - t0
    return ContractionResult(
        dense=np.asarray(dense, dtype=np.float64),
        contract_time_sec=t_contract,
        emit_time_sec=0.0,
        meta={"num_blocks": 1},
    )


def _cfg_signature(cfg: Any) -> Tuple[Tuple[str, object], ...]:
    names = (
        "tol",
        "tol_schedule",
        "tol_depth_decay",
        "tol_open_power",
        "randomized",
        "oversample",
        "n_power_iter",
        "implicit_merge_sketch",
        "optimize",
    )
    return tuple((name, getattr(cfg, name, None)) for name in names)


def _state_cache_key(part: PartitionNode, slice_map, cfg: Any) -> StateCacheKey:
    return (part.node_key, make_local_block_key(part.open_labels, slice_map), _cfg_signature(cfg))


def _rng_from_state_key(base_seed: int, state_key: StateCacheKey) -> np.random.Generator:
    payload = repr((int(base_seed), state_key)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    return np.random.default_rng(seed)


def _partition_depth_info(
    part: PartitionNode,
    depth: int = 0,
    output: Dict[tuple[int, ...], tuple[int, int, int]] | None = None,
) -> Dict[tuple[int, ...], tuple[int, int, int]]:
    if output is None:
        output = {}
    output[part.node_key] = (depth, len(part.open_labels), part.subtree_size)
    if part.children:
        for child in part.children:
            _partition_depth_info(child, depth + 1, output)
    return output


def _max_partition_depth(part: PartitionNode) -> int:
    if not part.children:
        return 0
    return 1 + max(_max_partition_depth(child) for child in part.children)


def _scheduled_tol(
    cfg: Any,
    *,
    base_tol: float,
    part: PartitionNode,
    depth_info: Dict[tuple[int, ...], tuple[int, int, int]],
    max_depth: int,
) -> float:
    schedule = str(getattr(cfg, "tol_schedule", "flat"))
    if schedule == "flat":
        return float(base_tol)
    depth, open_count, subtree_size = depth_info.get(part.node_key, (0, len(part.open_labels), part.subtree_size))
    depth_decay = float(getattr(cfg, "tol_depth_decay", 1.5))
    open_power = float(getattr(cfg, "tol_open_power", 0.5))
    depth_factor = depth_decay ** float(depth)
    open_factor = float(max(1, open_count)) ** float(open_power)
    tol = float(base_tol) * depth_factor / open_factor
    if schedule == "depth_open":
        return float(tol)
    if schedule == "depth_size_open":
        size_factor = float(max(1, subtree_size)) ** 0.25
        return float(tol / size_factor)
    if schedule == "root_strict":
        strictness = 1.0 + float(max_depth - depth)
        return float(base_tol / strictness)
    return float(base_tol)


def _leaf_state(
    tn: TensorNetwork,
    part: PartitionNode,
    slice_map,
    cfg,
    rng: np.random.Generator | None = None,
    stats: ASTNCRuntimeStats | None = None,
    local_tol: float | None = None,
) -> tuple[SeparatorState, float]:
    node_id = next(iter(part.node_ids))
    output_labels = list(part.open_labels) + list(part.boundary_labels)
    T = tn.contract_subnetwork([node_id], output_labels, slice_map=slice_map, optimize=getattr(cfg, "optimize", "greedy"))
    open_dims = [len(slice_map[label]) if label in slice_map else tn.label_dims[label] for label in part.open_labels]
    boundary_dims = [tn.label_dims[label] for label in part.boundary_labels]
    tol = float(local_tol if local_tol is not None else getattr(cfg, "tol", 1e-3))
    state, residual_ratio = adaptive_state_from_tensor(
        T,
        part.open_labels,
        open_dims,
        part.boundary_labels,
        boundary_dims,
        tol=tol,
        randomized=bool(getattr(cfg, "randomized", True)),
        oversample=int(getattr(cfg, "oversample", 4)),
        n_power_iter=int(getattr(cfg, "n_power_iter", 1)),
        rng=rng,
    )
    if stats is not None:
        open_flat = int(np.prod(open_dims)) if open_dims else 1
        boundary_flat = int(np.prod(boundary_dims)) if boundary_dims else 1
        full_rank = min(open_flat, boundary_flat)
        stats.observe_leaf_choice(compressed=bool(state.rank < full_rank))
    if stats is not None:
        stats.observe(is_leaf=True)
    return state, float(residual_ratio)


def _build_state(
    tn: TensorNetwork,
    part: PartitionNode,
    slice_map,
    cfg,
    *,
    cache: SeparatorStateCache | None = None,
    stats: ASTNCRuntimeStats | None = None,
    base_seed: int = 0,
    depth_info: Dict[tuple[int, ...], tuple[int, int, int]] | None = None,
    max_depth: int = 0,
    block_index: int = 0,
    tree_trace: ASTNCTreeTrace | None = None,
) -> SeparatorState:
    local_tol = float(getattr(cfg, "tol", 1e-3 if part.is_leaf else 1e-2))
    if depth_info is not None:
        local_tol = _scheduled_tol(
            cfg,
            base_tol=float(getattr(cfg, "tol", 1e-3 if part.is_leaf else 1e-2)),
            part=part,
            depth_info=depth_info,
            max_depth=max_depth,
        )
    state_key = _state_cache_key(part, slice_map, cfg)
    if cache is not None:
        cached = cache.get(state_key)
        if cached is not None:
            if tree_trace is not None:
                tree_trace.record_cache_hit(
                    state_key=state_key,
                    part=part,
                    block_index=block_index,
                    slice_map=slice_map,
                    state=cached,
                    actual_tol=local_tol,
                )
            return cached
    rng = _rng_from_state_key(base_seed, state_key)
    if part.is_leaf:
        state, leaf_residual_ratio = _leaf_state(tn, part, slice_map, cfg, rng=rng, stats=stats, local_tol=local_tol)
        if tree_trace is not None:
            open_dims = [len(slice_map[label]) if label in slice_map else tn.label_dims[label] for label in part.open_labels]
            boundary_dims = [tn.label_dims[label] for label in part.boundary_labels]
            tree_trace.record_leaf(
                state_key=state_key,
                part=part,
                block_index=block_index,
                slice_map=slice_map,
                state=state,
                actual_tol=local_tol,
                full_rank=min(
                    int(np.prod(open_dims)) if open_dims else 1,
                    int(np.prod(boundary_dims)) if boundary_dims else 1,
                ),
                residual_ratio=leaf_residual_ratio,
                source="computed",
            )
        if cache is not None:
            cache.put(state_key, state)
        return state

    left, right = part.children
    s_left = _build_state(
        tn,
        left,
        slice_map,
        cfg,
        cache=cache,
        stats=stats,
        base_seed=base_seed,
        depth_info=depth_info,
        max_depth=max_depth,
        block_index=block_index,
        tree_trace=tree_trace,
    )
    s_right = _build_state(
        tn,
        right,
        slice_map,
        cfg,
        cache=cache,
        stats=stats,
        base_seed=base_seed,
        depth_info=depth_info,
        max_depth=max_depth,
        block_index=block_index,
        tree_trace=tree_trace,
    )
    state, merge_info = merge_states(
        s_left,
        s_right,
        cut_labels=part.cut_labels,
        parent_boundary_labels=part.boundary_labels,
        label_dims=tn.label_dims,
        tol=local_tol,
        randomized=bool(getattr(cfg, "randomized", True)),
        oversample=int(getattr(cfg, "oversample", 4)),
        n_power_iter=int(getattr(cfg, "n_power_iter", 1)),
        implicit_merge_sketch=bool(getattr(cfg, "implicit_merge_sketch", True)),
        rng=rng,
    )
    if stats is not None:
        stats.observe_merge(merge_info)
        stats.observe(is_leaf=False)
    if tree_trace is not None:
        tree_trace.record_merge(
            state_key=state_key,
            part=part,
            block_index=block_index,
            slice_map=slice_map,
            state=state,
            actual_tol=local_tol,
            merge_info=merge_info,
            source="computed",
        )
    if cache is not None:
        cache.put(state_key, state)
    return state


def _state_to_dense(state: SeparatorState) -> np.ndarray:
    if len(state.boundary_labels) != 0:
        raise ValueError("Root state must have empty boundary.")
    if state.A.ndim == 1:
        return np.array(np.dot(state.A, state.B), dtype=np.float64)
    return np.tensordot(state.A, state.B, axes=([-1], [0]))


def astnc_contraction(
    tn: TensorNetwork,
    blocks: List[OutputBlock],
    cfg,
    cache: SeparatorStateCache | None = None,
) -> ContractionResult:
    import time

    partition = build_partition_tree(tn)
    depth_info = _partition_depth_info(partition)
    max_depth = _max_partition_depth(partition)
    state_cache = cache if cache is not None else SeparatorStateCache(enabled=bool(getattr(cfg, "cache_enabled", True)))
    stats = ASTNCRuntimeStats()
    tree_trace = ASTNCTreeTrace.build(partition, depth_info)
    base_seed = int(getattr(cfg, "seed", 0))
    dense = np.zeros(tn.output_shape, dtype=np.float64)
    t_contract = 0.0
    t_emit = 0.0
    block_labels = list(blocks[0].slice_map.keys()) if blocks else []
    suffix_labels = tn.open_label_order[len(block_labels):]

    for block_index, block in enumerate(blocks):
        tree_trace.register_block(block_index, block)
        t0 = time.perf_counter()
        state = _build_state(
            tn,
            partition,
            block.slice_map,
            cfg,
            cache=state_cache,
            stats=stats,
            base_seed=base_seed,
            depth_info=depth_info,
            max_depth=max_depth,
            block_index=block_index,
            tree_trace=tree_trace,
        )
        perm = [state.open_labels.index(label) for label in tn.open_label_order]
        block_tensor = _state_to_dense(state)
        if list(range(len(perm))) != perm:
            block_tensor = np.transpose(block_tensor, perm)
        t_contract += time.perf_counter() - t0
        if block_labels:
            row_slices = tuple(
                slice(min(indices), max(indices) + 1)
                for _, indices in sorted(block.slice_map.items(), key=lambda item: tn.open_label_order.index(item[0]))
            )
            target = row_slices + tuple(slice(None) for _ in suffix_labels)
        else:
            target = tuple(slice(None) for _ in tn.open_label_order)
        t1 = time.perf_counter()
        dense[target] = block_tensor
        t_emit += time.perf_counter() - t1

    return ContractionResult(
        dense=dense,
        contract_time_sec=t_contract,
        emit_time_sec=t_emit,
        meta={
            "num_blocks": len(blocks),
            **stats.summary(),
            **state_cache.summary(),
            "blocks": tree_trace.blocks,
            "tree": tree_trace.tree,
        },
    )
