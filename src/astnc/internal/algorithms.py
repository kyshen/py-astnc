from __future__ import annotations

import copy
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
    compressed_state_from_tensor,
    exact_state_from_tensor,
    merge_states,
)


@dataclass
class MaterializationResult:
    dense: np.ndarray
    contract_time_sec: float
    emit_time_sec: float
    meta: Dict[str, Any]


@dataclass
class ASTNCRuntimeStats:
    leaf_states_built: int = 0
    internal_states_built: int = 0
    peak_rank: int = 0
    num_exact_merges: int = 0
    num_compressed_merges: int = 0
    skipped_small_rank_merges: int = 0
    skipped_small_state_merges: int = 0
    skipped_low_saving_merges: int = 0
    num_implicit_merge_sketches: int = 0
    num_explicit_merge_compressions: int = 0
    num_exact_leaves: int = 0
    num_compressed_leaves: int = 0
    leaf_residual_ratio_sum: float = 0.0
    leaf_residual_ratio_count: int = 0
    merge_residual_ratio_sum: float = 0.0
    merge_residual_ratio_count: int = 0
    leaf_tol_sum: float = 0.0
    leaf_tol_count: int = 0
    merge_tol_sum: float = 0.0
    merge_tol_count: int = 0

    def observe(self, state: SeparatorState, *, is_leaf: bool) -> None:
        if is_leaf:
            self.leaf_states_built += 1
        else:
            self.internal_states_built += 1
        self.peak_rank = max(self.peak_rank, int(state.rank))

    def observe_leaf_choice(self, *, compressed: bool, residual_ratio: float) -> None:
        if compressed:
            self.num_compressed_leaves += 1
        else:
            self.num_exact_leaves += 1
        self.leaf_residual_ratio_sum += float(residual_ratio)
        self.leaf_residual_ratio_count += 1

    def observe_leaf_tol(self, tol: float) -> None:
        self.leaf_tol_sum += float(tol)
        self.leaf_tol_count += 1

    def observe_merge(self, merge_info: MergeInfo) -> None:
        self.merge_residual_ratio_sum += float(merge_info.residual_ratio)
        self.merge_residual_ratio_count += 1
        if merge_info.compressed:
            self.num_compressed_merges += 1
            if merge_info.path == "implicit_randomized":
                self.num_implicit_merge_sketches += 1
            else:
                self.num_explicit_merge_compressions += 1
            return
        self.num_exact_merges += 1
        if merge_info.reason == "rank_product_too_small":
            self.skipped_small_rank_merges += 1
        elif merge_info.reason == "exact_state_small_enough":
            self.skipped_small_state_merges += 1
        elif merge_info.reason == "saving_ratio_too_small":
            self.skipped_low_saving_merges += 1

    def observe_merge_tol(self, tol: float) -> None:
        self.merge_tol_sum += float(tol)
        self.merge_tol_count += 1

    def summary(self) -> Dict[str, int | float]:
        return {
            "leaf_states_built": int(self.leaf_states_built),
            "internal_states_built": int(self.internal_states_built),
            "peak_rank": int(self.peak_rank),
            "num_exact_merges": int(self.num_exact_merges),
            "num_compressed_merges": int(self.num_compressed_merges),
            "skipped_small_rank_merges": int(self.skipped_small_rank_merges),
            "skipped_small_state_merges": int(self.skipped_small_state_merges),
            "skipped_low_saving_merges": int(self.skipped_low_saving_merges),
            "num_implicit_merge_sketches": int(self.num_implicit_merge_sketches),
            "num_explicit_merge_compressions": int(self.num_explicit_merge_compressions),
            "num_exact_leaves": int(self.num_exact_leaves),
            "num_compressed_leaves": int(self.num_compressed_leaves),
            "mean_leaf_residual_ratio": float(self.leaf_residual_ratio_sum / self.leaf_residual_ratio_count) if self.leaf_residual_ratio_count else 0.0,
            "mean_merge_residual_ratio": float(self.merge_residual_ratio_sum / self.merge_residual_ratio_count) if self.merge_residual_ratio_count else 0.0,
            "mean_leaf_tol": float(self.leaf_tol_sum / self.leaf_tol_count) if self.leaf_tol_count else 0.0,
            "mean_merge_tol": float(self.merge_tol_sum / self.merge_tol_count) if self.merge_tol_count else 0.0,
        }


def materialize_exact(tn: TensorNetwork, blocks: List[OutputBlock], optimize: str = "optimal") -> MaterializationResult:
    import time

    dense = np.zeros(tn.output_shape, dtype=np.float64)
    t_contract = 0.0
    t_emit = 0.0
    block_labels = list(blocks[0].slice_map.keys()) if blocks else []
    suffix_labels = tn.open_label_order[len(block_labels):]
    for block in blocks:
        t0 = time.perf_counter()
        block_tensor = tn.contract_full(slice_map=block.slice_map, optimize=optimize)
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
    return MaterializationResult(
        dense=dense,
        contract_time_sec=t_contract,
        emit_time_sec=t_emit,
        meta={"num_blocks": len(blocks)},
    )


def _cfg_signature(cfg: Any) -> Tuple[Tuple[str, object], ...]:
    names = (
        "rank_policy",
        "leaf_tol",
        "merge_tol",
        "tol_schedule",
        "tol_depth_decay",
        "tol_open_power",
        "target_rank",
        "max_rank",
        "randomized",
        "oversample",
        "n_power_iter",
        "selective_threshold",
        "compress_min_rank_product",
        "compress_max_exact_size",
        "compress_min_saving_ratio",
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
    local_leaf_tol: float | None = None,
) -> SeparatorState:
    node_id = next(iter(part.node_ids))
    output_labels = list(part.open_labels) + list(part.boundary_labels)
    T = tn.contract_subnetwork([node_id], output_labels, slice_map=slice_map, optimize=getattr(cfg, "optimize", "greedy"))
    open_dims = [len(slice_map[label]) if label in slice_map else tn.label_dims[label] for label in part.open_labels]
    boundary_dims = [tn.label_dims[label] for label in part.boundary_labels]
    rank_policy = str(getattr(cfg, "rank_policy", "fixed"))
    if rank_policy == "adaptive":
        tol = float(local_leaf_tol if local_leaf_tol is not None else getattr(cfg, "leaf_tol", 1e-3))
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
            stats.observe_leaf_tol(tol)
            stats.observe_leaf_choice(compressed=bool(state.rank < full_rank), residual_ratio=float(residual_ratio))
    elif getattr(cfg, "target_rank", 0) <= 0:
        state = exact_state_from_tensor(T, part.open_labels, open_dims, part.boundary_labels, boundary_dims)
        if stats is not None:
            stats.observe_leaf_choice(compressed=False, residual_ratio=0.0)
    else:
        target_rank = min(
            int(getattr(cfg, "target_rank", 1)),
            max(1, int(np.prod(open_dims)) if open_dims else 1),
            max(1, int(np.prod(boundary_dims)) if boundary_dims else 1),
        )
        state = compressed_state_from_tensor(
            T,
            part.open_labels,
            open_dims,
            part.boundary_labels,
            boundary_dims,
            target_rank=target_rank,
            randomized=bool(getattr(cfg, "randomized", True)),
            oversample=int(getattr(cfg, "oversample", 4)),
            n_power_iter=int(getattr(cfg, "n_power_iter", 1)),
            rng=rng,
        )
        if stats is not None:
            open_flat = int(np.prod(open_dims)) if open_dims else 1
            boundary_flat = int(np.prod(boundary_dims)) if boundary_dims else 1
            full_rank = min(open_flat, boundary_flat)
            stats.observe_leaf_choice(compressed=bool(state.rank < full_rank), residual_ratio=0.0)
    if stats is not None:
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
    depth_info: Dict[tuple[int, ...], tuple[int, int, int]] | None = None,
    max_depth: int = 0,
) -> SeparatorState:
    state_key = _state_cache_key(part, slice_map, cfg)
    if cache is not None:
        cached = cache.get(state_key)
        if cached is not None:
            return cached
    rng = _rng_from_state_key(base_seed, state_key)
    if part.is_leaf:
        local_leaf_tol = None
        if str(getattr(cfg, "rank_policy", "fixed")) == "adaptive" and depth_info is not None:
            local_leaf_tol = _scheduled_tol(
                cfg,
                base_tol=float(getattr(cfg, "leaf_tol", 1e-3)),
                part=part,
                depth_info=depth_info,
                max_depth=max_depth,
            )
        state = _leaf_state(tn, part, slice_map, cfg, rng=rng, stats=stats, local_leaf_tol=local_leaf_tol)
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
    )
    rank_policy = str(getattr(cfg, "rank_policy", "fixed"))
    rank = min(int(getattr(cfg, "target_rank", 1)), int(getattr(cfg, "max_rank", getattr(cfg, "target_rank", 1))))
    local_merge_tol = float(getattr(cfg, "merge_tol", 1e-2))
    if rank_policy == "adaptive" and depth_info is not None:
        local_merge_tol = _scheduled_tol(
            cfg,
            base_tol=float(getattr(cfg, "merge_tol", 1e-2)),
            part=part,
            depth_info=depth_info,
            max_depth=max_depth,
        )
        if stats is not None:
            stats.observe_merge_tol(local_merge_tol)
    state, merge_info = merge_states(
        s_left,
        s_right,
        cut_labels=part.cut_labels,
        parent_boundary_labels=part.boundary_labels,
        label_dims=tn.label_dims,
        target_rank=rank,
        rank_policy=rank_policy,
        merge_tol=local_merge_tol,
        randomized=bool(getattr(cfg, "randomized", True)),
        oversample=int(getattr(cfg, "oversample", 4)),
        n_power_iter=int(getattr(cfg, "n_power_iter", 1)),
        selective_threshold=int(getattr(cfg, "selective_threshold", 0)),
        compress_min_rank_product=int(getattr(cfg, "compress_min_rank_product", 0)),
        compress_max_exact_size=int(getattr(cfg, "compress_max_exact_size", 0)),
        compress_min_saving_ratio=float(getattr(cfg, "compress_min_saving_ratio", 0.0)),
        implicit_merge_sketch=bool(getattr(cfg, "implicit_merge_sketch", True)),
        implicit_min_full_rank=int(getattr(cfg, "implicit_min_full_rank", 192)),
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


def _maybe_refine_block(
    tn: TensorNetwork,
    block: OutputBlock,
    approx_block: np.ndarray,
    cfg,
    cache: SeparatorStateCache | None = None,
    stats: ASTNCRuntimeStats | None = None,
    base_seed: int = 0,
    depth_info: Dict[tuple[int, ...], tuple[int, int, int]] | None = None,
    max_depth: int = 0,
) -> tuple[np.ndarray, int]:
    if not bool(getattr(cfg, "adaptive_refine", False)):
        if str(getattr(cfg, "rank_policy", "fixed")) == "adaptive":
            return approx_block, -1
        return approx_block, int(getattr(cfg, "target_rank", 1))
    target_rank = int(getattr(cfg, "target_rank", 1))
    tol = float(getattr(cfg, "refine_tol", 1e-3))
    max_steps = int(getattr(cfg, "max_refine_steps", 0))
    max_rank = int(getattr(cfg, "max_rank", target_rank))
    growth = int(getattr(cfg, "rank_growth_factor", 2))
    refine_steps = 0
    current = approx_block
    while refine_steps < max_steps:
        reference = tn.contract_full(slice_map=block.slice_map, optimize="greedy")
        err = float(np.linalg.norm(reference - current) / (np.linalg.norm(reference) + 1e-12))
        if err <= tol or target_rank >= max_rank:
            return current, target_rank
        target_rank = min(max_rank, max(target_rank + 1, target_rank * growth))
        refine_steps += 1
        local_cfg = copy.deepcopy(cfg)
        setattr(local_cfg, "target_rank", target_rank)
        partition = build_partition_tree(tn)
        state = _build_state(
            tn,
            partition,
            block.slice_map,
            local_cfg,
            cache=cache,
            stats=stats,
            base_seed=base_seed,
            depth_info=depth_info,
            max_depth=max_depth,
        )
        current = _state_to_dense(state)
    return current, target_rank


def materialize_astnc(
    tn: TensorNetwork,
    blocks: List[OutputBlock],
    cfg,
    cache: SeparatorStateCache | None = None,
) -> MaterializationResult:
    import time

    partition = build_partition_tree(tn)
    depth_info = _partition_depth_info(partition)
    max_depth = _max_partition_depth(partition)
    state_cache = cache if cache is not None else SeparatorStateCache(enabled=bool(getattr(cfg, "cache_enabled", True)))
    stats = ASTNCRuntimeStats()
    base_seed = int(getattr(cfg, "seed", 0))
    dense = np.zeros(tn.output_shape, dtype=np.float64)
    t_contract = 0.0
    t_emit = 0.0
    refined_blocks = 0
    used_ranks: List[int] = []
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
            depth_info=depth_info,
            max_depth=max_depth,
        )
        perm = [state.open_labels.index(label) for label in tn.open_label_order]
        block_tensor = _state_to_dense(state)
        current_rank = int(state.rank)
        if list(range(len(perm))) != perm:
            block_tensor = np.transpose(block_tensor, perm)
        if bool(getattr(cfg, "adaptive_refine", False)):
            old_rank = current_rank if str(getattr(cfg, "rank_policy", "fixed")) == "adaptive" else int(getattr(cfg, "target_rank", 1))
            block_tensor, used_rank = _maybe_refine_block(
                tn,
                block,
                block_tensor,
                cfg,
                cache=state_cache,
                stats=stats,
                base_seed=base_seed,
                depth_info=depth_info,
                max_depth=max_depth,
            )
            if used_rank != old_rank:
                refined_blocks += 1
            used_ranks.append(int(current_rank if used_rank < 0 else used_rank))
        else:
            used_ranks.append(int(current_rank))
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

    return MaterializationResult(
        dense=dense,
        contract_time_sec=t_contract,
        emit_time_sec=t_emit,
        meta={
            "num_blocks": len(blocks),
            "refined_blocks": refined_blocks,
            "mean_rank": float(np.mean(used_ranks)) if used_ranks else 0.0,
            "max_rank": max(used_ranks) if used_ranks else 0,
            **stats.summary(),
            **state_cache.summary(),
        },
    )
