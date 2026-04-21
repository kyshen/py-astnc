from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from astnc.internal.algorithms import astnc_contraction, exact_contraction
from astnc.internal.blocking import make_blocks
from astnc.types import ContractionCache
from astnc.workpoints import get_workpoint


@dataclass
class ASTNCConfig:
    optimize: str = "greedy"
    seed: int = 0
    leaf_tol: float = 1e-3
    merge_tol: float = 5e-3
    tol_schedule: str = "depth_open"
    tol_depth_decay: float = 1.5
    tol_open_power: float = 0.5
    randomized: bool = True
    oversample: int = 1
    n_power_iter: int = 0
    implicit_merge_sketch: bool = True
    cache_enabled: bool = True


def _astnc_config_from_options(workpoint: str | None, options: Dict[str, Any] | None) -> ASTNCConfig:
    cfg = ASTNCConfig()
    if workpoint is not None:
        for key, value in get_workpoint(workpoint).method_options.items():
            setattr(cfg, key, value)
    if options:
        for key, value in options.items():
            setattr(cfg, key, value)
    return cfg


def contract_exact(
    tn,
    *,
    optimize: str = "optimal",
    return_info: bool = False,
):
    result = exact_contraction(tn, optimize=str(optimize))
    info = {
        "method": "exact",
        "workpoint": None,
        "num_blocks": 1,
        "block_spec": {},
        "output_shape": tuple(int(dim) for dim in tn.output_shape),
        "contract_time_sec": float(result.contract_time_sec),
        "emit_time_sec": float(result.emit_time_sec),
        "total_time_sec": float(result.contract_time_sec + result.emit_time_sec),
        "rel_method": "exact",
        "meta": dict(result.meta),
    }
    if return_info:
        return np.asarray(result.dense), info
    return np.asarray(result.dense)


def contract_astnc(
    tn,
    *,
    workpoint: str | None = None,
    block_spec: Mapping[int, int] | None = None,
    cache: ContractionCache | None = None,
    return_info: bool = False,
    **options: Any,
):
    cfg = _astnc_config_from_options(workpoint, options or None)
    resolved_block_spec = {int(axis): int(size) for axis, size in dict(block_spec or {}).items()}
    blocks = make_blocks(tn, block_spec=resolved_block_spec)
    state_cache = None if cache is None else cache.state_cache
    if state_cache is not None:
        state_cache.enabled = bool(cfg.cache_enabled)
    result = astnc_contraction(tn, blocks, cfg, cache=state_cache)
    info = {
        "method": "astnc",
        "workpoint": workpoint,
        "num_blocks": len(blocks),
        "block_spec": dict(resolved_block_spec),
        "output_shape": tuple(int(dim) for dim in tn.output_shape),
        "contract_time_sec": float(result.contract_time_sec),
        "emit_time_sec": float(result.emit_time_sec),
        "total_time_sec": float(result.contract_time_sec + result.emit_time_sec),
        "rel_method": "astnc",
        "meta": dict(result.meta),
    }
    if return_info:
        return np.asarray(result.dense), info
    return np.asarray(result.dense)


def create_contraction_cache(*, enabled: bool = True) -> ContractionCache:
    cache = ContractionCache()
    cache.state_cache.enabled = bool(enabled)
    return cache
