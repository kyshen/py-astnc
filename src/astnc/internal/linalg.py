from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LowRankFactors:
    left: np.ndarray
    right: np.ndarray


@dataclass
class AdaptiveCompressionResult:
    factors: LowRankFactors
    chosen_rank: int
    residual_ratio: float


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def orth(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    q, _ = np.linalg.qr(x, mode="reduced")
    return q


def choose_rank_from_singular_values(s: np.ndarray, tol: float) -> tuple[int, float]:
    if len(s) == 0:
        return 0, 0.0
    if tol <= 0:
        return len(s), 0.0
    sq = np.square(np.asarray(s, dtype=float))
    total = float(np.sum(sq))
    if total <= 1e-24:
        return 1, 0.0
    cumulative = np.cumsum(sq)
    tail_sq = total - cumulative
    threshold = float(tol) ** 2 * total
    for idx, rem_sq in enumerate(tail_sq, start=1):
        if rem_sq <= threshold:
            return idx, float(np.sqrt(max(rem_sq, 0.0) / total))
    return len(s), 0.0


def compress_matrix_adaptive(
    M: np.ndarray,
    tol: float,
    randomized: bool = True,
    oversample: int = 4,
    n_power_iter: int = 1,
    rng: np.random.Generator | None = None,
) -> AdaptiveCompressionResult:
    del randomized, oversample, n_power_iter, rng
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    rank, residual_ratio = choose_rank_from_singular_values(s, tol)
    rank = max(1, min(rank, len(s)))
    factors = LowRankFactors(left=U[:, :rank] * s[:rank], right=Vt[:rank, :].T)
    return AdaptiveCompressionResult(factors=factors, chosen_rank=rank, residual_ratio=residual_ratio)


def compress_from_factors_adaptive(
    A: np.ndarray,
    B: np.ndarray,
    tol: float,
    randomized: bool = True,
    oversample: int = 4,
    n_power_iter: int = 1,
    rng: np.random.Generator | None = None,
) -> AdaptiveCompressionResult:
    del randomized, oversample, n_power_iter, rng
    Qa, Ra = np.linalg.qr(A, mode="reduced")
    Qb, Rb = np.linalg.qr(B, mode="reduced")
    core = Ra @ Rb.T
    U, s, Vt = np.linalg.svd(core, full_matrices=False)
    rank, residual_ratio = choose_rank_from_singular_values(s, tol)
    rank = max(1, min(rank, len(s)))
    left = Qa @ U[:, :rank] * s[:rank]
    right = Qb @ Vt[:rank, :].T
    return AdaptiveCompressionResult(
        factors=LowRankFactors(left=left, right=right),
        chosen_rank=rank,
        residual_ratio=residual_ratio,
    )


def compress_from_implicit_factors_adaptive(
    *,
    num_rows: int,
    num_cols: int,
    latent_rank: int,
    apply_A,
    apply_AT,
    apply_B,
    apply_BT,
    tol: float,
    oversample: int = 4,
    n_power_iter: int = 1,
    rng: np.random.Generator | None = None,
) -> AdaptiveCompressionResult:
    if latent_rank <= 0:
        raise ValueError("Implicit adaptive compression expects a positive latent rank.")
    rng = _resolve_rng(rng)
    max_subspace = min(int(latent_rank), int(num_rows), int(num_cols))
    sketch_cols = min(max_subspace, max(int(oversample) + 1, 2))
    while True:
        omega = rng.standard_normal((int(num_cols), sketch_cols))
        Y = apply_A(apply_BT(omega))
        for _ in range(max(0, int(n_power_iter))):
            Y = apply_A(apply_BT(apply_B(apply_AT(Y))))
        Q = orth(Y)
        small = apply_B(apply_AT(Q)).T
        Uh, s, Vt = np.linalg.svd(small, full_matrices=False)
        rank, residual_ratio = choose_rank_from_singular_values(s, tol)
        rank = max(1, min(rank, len(s)))
        enough_room = rank + int(oversample) + 1 <= sketch_cols
        if enough_room or sketch_cols >= max_subspace:
            U = Q @ Uh[:, :rank]
            return AdaptiveCompressionResult(
                factors=LowRankFactors(left=U * s[:rank], right=Vt[:rank, :].T),
                chosen_rank=rank,
                residual_ratio=residual_ratio,
            )
        next_cols = min(max_subspace, max(sketch_cols * 2, rank + int(oversample) + 1))
        if next_cols == sketch_cols:
            U = Q @ Uh[:, :rank]
            return AdaptiveCompressionResult(
                factors=LowRankFactors(left=U * s[:rank], right=Vt[:rank, :].T),
                chosen_rank=rank,
                residual_ratio=residual_ratio,
            )
        sketch_cols = next_cols
