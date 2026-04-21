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
    internal_rank_sum: int = 0
    internal_rank_count: int = 0
    max_internal_rank: int = 0
    merge_residual_ratio_sum: float = 0.0
    merge_residual_ratio_count: int = 0
    tol_sum: float = 0.0
    tol_count: int = 0

    def observe(self, state: SeparatorState, *, is_leaf: bool) -> None:
        if is_leaf:
            self.leaf_states_built += 1
        else:
            self.internal_states_built += 1
            if len(state.boundary_labels) != 0:
                rank = int(state.rank)
                self.internal_rank_sum += rank
                self.internal_rank_count += 1
                self.max_internal_rank = max(self.max_internal_rank, rank)

    def observe_tol(self, tol: float) -> None:
        self.tol_sum += float(tol)
        self.tol_count += 1

    def observe_merge(self, merge_info: MergeInfo) -> None:
        self.merge_residual_ratio_sum += float(merge_info.residual_ratio)
        self.merge_residual_ratio_count += 1

    def summary(self) -> Dict[str, int | float]:
        return {
            "leaf_states_built": int(self.leaf_states_built),
            "internal_states_built": int(self.internal_states_built),
            "mean_internal_rank": float(self.internal_rank_sum / self.internal_rank_count) if self.internal_rank_count else 0.0,
            "max_internal_rank": int(self.max_internal_rank),
            "mean_merge_residual_ratio": float(self.merge_residual_ratio_sum / self.merge_residual_ratio_count) if self.merge_residual_ratio_count else 0.0,
            "mean_tol": float(self.tol_sum / self.tol_count) if self.tol_count else 0.0,
        }


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


def _scheduled_tol(
    cfg: Any,
    *,
    base_tol: float,
    part: PartitionNode,
    tree_height: int,
) -> float:
    schedule = str(getattr(cfg, "tol_schedule", "flat"))
    if schedule == "flat":
        return float(base_tol)
    depth = int(part.depth)
    open_count = len(part.open_labels)
    subtree_size = part.subtree_size
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
        strictness = 1.0 + float(tree_height - depth)
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
) -> SeparatorState:
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
        stats.observe_tol(tol)
        stats.observe(state, is_leaf=True)
    return state


def _build_state(
    tn: TensorNetwork,
    part: PartitionNode,
    slice_map,
    cfg,
    *,
    cache: SeparatorStateCache | None = None,
    stats: ASTNCRuntimeStats | None = None,
    base_seed: int = 0,
    tree_height: int = 0,
) -> SeparatorState:
    state_key = _state_cache_key(part, slice_map, cfg)
    if cache is not None:
        cached = cache.get(state_key)
        if cached is not None:
            return cached
    rng = _rng_from_state_key(base_seed, state_key)
    if part.is_leaf:
        local_tol = None
        local_tol = _scheduled_tol(
            cfg,
            base_tol=float(getattr(cfg, "tol", 1e-3)),
            part=part,
            tree_height=tree_height,
        )
        state = _leaf_state(tn, part, slice_map, cfg, rng=rng, stats=stats, local_tol=local_tol)
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
        tree_height=tree_height,
    )
    s_right = _build_state(
        tn,
        right,
        slice_map,
        cfg,
        cache=cache,
        stats=stats,
        base_seed=base_seed,
        tree_height=tree_height,
    )
    local_tol = float(getattr(cfg, "tol", 1e-2))
    local_tol = _scheduled_tol(
        cfg,
        base_tol=float(getattr(cfg, "tol", 1e-2)),
        part=part,
        tree_height=tree_height,
    )
    if stats is not None:
        stats.observe_tol(local_tol)
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
        stats.observe(state, is_leaf=False)
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
    tree_height = partition.height
    state_cache = cache if cache is not None else SeparatorStateCache(enabled=bool(getattr(cfg, "cache_enabled", True)))
    stats = ASTNCRuntimeStats()
    base_seed = int(getattr(cfg, "seed", 0))
    dense = np.zeros(tn.output_shape, dtype=np.float64)
    t_contract = 0.0
    t_emit = 0.0
    block_labels = list(blocks[0].slice_map.keys()) if blocks else []
    suffix_labels = tn.open_label_order[len(block_labels):]

    for block in blocks:
        t0 = time.perf_counter()
        state = _build_state(
            tn,
            partition,
            block.slice_map,
            cfg,
            cache=state_cache,
            stats=stats,
            base_seed=base_seed,
            tree_height=tree_height,
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
        },
    )
