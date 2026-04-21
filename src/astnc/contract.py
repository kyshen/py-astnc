from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from astnc.internal.algorithms import astnc_contraction, exact_contraction
from astnc.internal.blocking import make_blocks


@dataclass
class ASTNCConfig:
    optimize: str = "greedy"
    seed: int = 0
    tol: float = 1e-3
    randomized: bool = True
    oversample: int = 1
    n_power_iter: int = 0
    implicit_merge_sketch: bool = True
    cache_enabled: bool = True


def contract_exact(
    tn,
    *,
    optimize: str = "optimal",
    return_info: bool = False,
):
    result = exact_contraction(tn, optimize=str(optimize))
    info = {
        "method": "exact",
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
    block_spec: Mapping[int, int] | None = None,
    return_info: bool = False,
    **options: Any,
):
    cfg = ASTNCConfig()
    for key, value in options.items():
        setattr(cfg, key, value)
    resolved_block_spec = {int(axis): int(size) for axis, size in dict(block_spec or {}).items()}
    blocks = make_blocks(tn, block_spec=resolved_block_spec)
    result = astnc_contraction(tn, blocks, cfg)
    info = {
        "method": "astnc",
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
