from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from astnc.errors import InvalidMethodError
from astnc.internal.algorithms import materialize_astnc, materialize_exact
from astnc.internal.blocking import make_blocks
from astnc.types import MaterializationCache
from astnc.workpoints import get_workpoint


@dataclass
class MethodConfig:
    optimize: str = "greedy"
    seed: int = 0
    rank_policy: str = "adaptive"
    leaf_tol: float = 1e-3
    merge_tol: float = 5e-3
    tol_schedule: str = "depth_open"
    tol_depth_decay: float = 1.5
    tol_open_power: float = 0.5
    target_rank: int = 2
    max_rank: int = 16
    randomized: bool = True
    oversample: int = 1
    n_power_iter: int = 0
    selective_threshold: int = 0
    compress_min_rank_product: int = 4
    compress_max_exact_size: int = 256
    compress_min_saving_ratio: float = 0.1
    implicit_merge_sketch: bool = True
    implicit_min_full_rank: int = 192
    adaptive_refine: bool = False
    refine_tol: float = 1e-3
    max_refine_steps: int = 0
    rank_growth_factor: int = 2
    cache_enabled: bool = True


def _config_from_options(method: str, workpoint: str | None, options: Dict[str, Any] | None) -> MethodConfig:
    cfg = MethodConfig()
    if method == "exact":
        cfg.optimize = "optimal"
        cfg.rank_policy = "fixed"
        cfg.target_rank = 0
        cfg.max_rank = 0
        cfg.cache_enabled = False
    elif method == "fixed_rank":
        cfg.rank_policy = "fixed"
        cfg.leaf_tol = 0.0
        cfg.merge_tol = 0.0
    elif method == "astnc":
        if workpoint is not None:
            for key, value in get_workpoint(workpoint).method_options.items():
                setattr(cfg, key, value)
    else:
        raise InvalidMethodError(f"Unknown method: {method!r}")
    if options:
        for key, value in options.items():
            setattr(cfg, key, value)
    if method == "fixed_rank":
        cfg.max_rank = max(int(cfg.max_rank), int(cfg.target_rank))
    return cfg


def materialize(
    tn,
    *,
    method: str = "astnc",
    workpoint: str | None = None,
    block_labels: int = 0,
    chunk_size: int = 1,
    cache: MaterializationCache | None = None,
    return_info: bool = False,
    **options: Any,
):
    cfg = _config_from_options(method, workpoint, options or None)
    blocks = make_blocks(tn, block_label_count=block_labels, chunk_size=chunk_size)
    if method == "exact":
        result = materialize_exact(tn, blocks, optimize=str(cfg.optimize))
    else:
        state_cache = None if cache is None else cache.state_cache
        if state_cache is not None:
            state_cache.enabled = bool(cfg.cache_enabled)
        result = materialize_astnc(tn, blocks, cfg, cache=state_cache)
    info = {
        "method": method,
        "workpoint": workpoint,
        "num_blocks": len(blocks),
        "block_labels": int(block_labels),
        "chunk_size": int(chunk_size),
        "output_shape": tuple(int(dim) for dim in tn.output_shape),
        "contract_time_sec": float(result.contract_time_sec),
        "emit_time_sec": float(result.emit_time_sec),
        "total_time_sec": float(result.contract_time_sec + result.emit_time_sec),
        "rel_method": "exact" if method == "exact" else method,
        "meta": dict(result.meta),
    }
    if return_info:
        return np.asarray(result.dense), info
    return np.asarray(result.dense)


def create_cache(*, enabled: bool = True) -> MaterializationCache:
    cache = MaterializationCache()
    cache.state_cache.enabled = bool(enabled)
    return cache
