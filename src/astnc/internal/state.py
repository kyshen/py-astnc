from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import opt_einsum as oe

from astnc.internal.linalg import (
    compress_from_factors_adaptive,
    compress_from_implicit_factors_adaptive,
    compress_matrix_adaptive,
)


@dataclass
class SeparatorState:
    open_labels: List[int]
    open_dims: List[int]
    boundary_labels: List[int]
    boundary_dims: List[int]
    A: np.ndarray
    B: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.A.shape[-1])

    def to_matrix(self) -> Tuple[np.ndarray, List[int], List[int]]:
        open_flat = int(np.prod(self.open_dims)) if self.open_dims else 1
        boundary_flat = int(np.prod(self.boundary_dims)) if self.boundary_dims else 1
        A = self.A.reshape(open_flat, self.rank)
        B = self.B.reshape(boundary_flat, self.rank)
        return A @ B.T, self.open_labels, self.boundary_labels


@dataclass
class MergeInfo:
    compressed: bool
    full_rank: int
    used_rank: int
    path: str = "explicit"
    residual_ratio: float = 0.0


def adaptive_state_from_tensor(
    T: np.ndarray,
    open_labels: Sequence[int],
    open_dims: Sequence[int],
    boundary_labels: Sequence[int],
    boundary_dims: Sequence[int],
    tol: float,
    randomized: bool = True,
    oversample: int = 4,
    n_power_iter: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[SeparatorState, float]:
    open_flat = int(np.prod(open_dims)) if open_dims else 1
    boundary_flat = int(np.prod(boundary_dims)) if boundary_dims else 1
    M = T.reshape(open_flat, boundary_flat)
    result = compress_matrix_adaptive(
        M,
        tol=tol,
        randomized=randomized,
        oversample=oversample,
        n_power_iter=n_power_iter,
        rng=rng,
    )
    rank = result.factors.left.shape[-1]
    A = result.factors.left.reshape(*open_dims, rank) if open_dims else result.factors.left.reshape(rank)
    B = result.factors.right.reshape(*boundary_dims, rank) if boundary_dims else result.factors.right.reshape(rank)
    state = SeparatorState(list(open_labels), list(open_dims), list(boundary_labels), list(boundary_dims), A, B)
    return state, float(result.residual_ratio)


def _make_merge_linear_ops(
    left: SeparatorState,
    right: SeparatorState,
    *,
    left_args,
    right_args,
    parent_boundary_labels: Sequence[int],
    parent_boundary_dims: Sequence[int],
    rank_label_left: int,
    rank_label_right: int,
):
    open_flat_left = int(np.prod(left.open_dims)) if left.open_dims else 1
    open_flat_right = int(np.prod(right.open_dims)) if right.open_dims else 1
    boundary_flat = int(np.prod(parent_boundary_dims)) if parent_boundary_dims else 1
    left_open = left.A.reshape(open_flat_left, left.rank)
    right_open = right.A.reshape(open_flat_right, right.rank)
    sample_label = -3

    def apply_A(z: np.ndarray) -> np.ndarray:
        z3 = z.reshape(left.rank, right.rank, -1)
        tmp = np.tensordot(left_open, z3, axes=([1], [0]))
        out = np.tensordot(tmp, right_open, axes=([1], [1]))
        return np.transpose(out, (0, 2, 1)).reshape(open_flat_left * open_flat_right, -1)

    def apply_AT(y: np.ndarray) -> np.ndarray:
        y3 = y.reshape(open_flat_left, open_flat_right, -1)
        tmp = np.tensordot(y3, right_open, axes=([1], [0]))
        tmp = np.transpose(tmp, (0, 2, 1))
        out = np.tensordot(left_open.T, tmp, axes=([1], [0]))
        return out.reshape(left.rank * right.rank, -1)

    def apply_B(z: np.ndarray) -> np.ndarray:
        z3 = z.reshape(left.rank, right.rank, -1)
        res = oe.contract(
            *left_args,
            *right_args,
            z3,
            [rank_label_left, rank_label_right, sample_label],
            list(parent_boundary_labels) + [sample_label],
            optimize="greedy",
        )
        return res.reshape(boundary_flat, -1)

    def apply_BT(omega: np.ndarray) -> np.ndarray:
        if parent_boundary_dims:
            omega_tensor = omega.reshape(*parent_boundary_dims, -1)
            omega_labels = list(parent_boundary_labels) + [sample_label]
        else:
            omega_tensor = omega.reshape(-1)
            omega_labels = [sample_label]
        res = oe.contract(
            *left_args,
            *right_args,
            omega_tensor,
            omega_labels,
            [rank_label_left, rank_label_right, sample_label],
            optimize="greedy",
        )
        return res.reshape(left.rank * right.rank, -1)

    return apply_A, apply_AT, apply_B, apply_BT


def merge_states(
    left: SeparatorState,
    right: SeparatorState,
    cut_labels: Sequence[int],
    parent_boundary_labels: Sequence[int],
    label_dims: Dict[int, int],
    merge_tol: float = 1e-2,
    randomized: bool = True,
    oversample: int = 4,
    n_power_iter: int = 1,
    implicit_merge_sketch: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[SeparatorState, MergeInfo]:
    open_labels = list(left.open_labels) + list(right.open_labels)
    open_dims = list(left.open_dims) + list(right.open_dims)
    parent_boundary_labels = list(parent_boundary_labels)
    parent_boundary_dims = [label_dims[label] for label in parent_boundary_labels]

    rL = left.rank
    rR = right.rank
    open_flat = int(np.prod(open_dims)) if open_dims else 1
    B_L = left.B.reshape(rL) if len(left.boundary_labels) == 0 else left.B
    B_R = right.B.reshape(rR) if len(right.boundary_labels) == 0 else right.B
    rank_label_left = -1
    rank_label_right = -2
    left_labels = [-(100000 + i) for i in range(len(left.boundary_labels))] + [rank_label_left]
    right_labels = [-(200000 + i) for i in range(len(right.boundary_labels))] + [rank_label_right]
    for idx, label in enumerate(left.boundary_labels):
        if label in cut_labels or label in parent_boundary_labels:
            left_labels[idx] = label
    for idx, label in enumerate(right.boundary_labels):
        if label in cut_labels or label in parent_boundary_labels:
            right_labels[idx] = label
    output_labels = list(parent_boundary_labels) + [rank_label_left, rank_label_right]
    left_args = [B_L, [rank_label_left]] if len(left.boundary_labels) == 0 else [B_L, left_labels]
    right_args = [B_R, [rank_label_right]] if len(right.boundary_labels) == 0 else [B_R, right_labels]
    boundary_flat = int(np.prod(parent_boundary_dims)) if parent_boundary_dims else 1
    full_rank = int(rL * rR)
    merge_info = MergeInfo(compressed=True, full_rank=int(full_rank), used_rank=int(full_rank))
    use_implicit_path = bool(
        randomized
        and bool(implicit_merge_sketch)
        and min(open_flat, boundary_flat) > 1
        and full_rank > 1
    )
    if use_implicit_path:
        apply_A, apply_AT, apply_B, apply_BT = _make_merge_linear_ops(
            left,
            right,
            left_args=left_args,
            right_args=right_args,
            parent_boundary_labels=parent_boundary_labels,
            parent_boundary_dims=parent_boundary_dims,
            rank_label_left=rank_label_left,
            rank_label_right=rank_label_right,
        )
        result = compress_from_implicit_factors_adaptive(
            num_rows=open_flat,
            num_cols=boundary_flat,
            latent_rank=full_rank,
            apply_A=apply_A,
            apply_AT=apply_AT,
            apply_B=apply_B,
            apply_BT=apply_BT,
            tol=float(merge_tol),
            oversample=oversample,
            n_power_iter=n_power_iter,
            rng=rng,
        )
        factors = result.factors
        merge_info.used_rank = int(result.chosen_rank)
        merge_info.compressed = bool(result.chosen_rank < full_rank)
        merge_info.path = "implicit_randomized"
        merge_info.residual_ratio = float(result.residual_ratio)
    else:
        A_outer = np.tensordot(left.A, right.A, axes=0)
        num_open_left = len(left.open_dims)
        num_open_right = len(right.open_dims)
        perm = list(range(num_open_left)) + list(range(num_open_left + 1, num_open_left + 1 + num_open_right)) + [num_open_left, num_open_left + 1 + num_open_right]
        A_outer = np.transpose(A_outer, perm)
        A_mat = A_outer.reshape(open_flat, full_rank)
        C = oe.contract(*left_args, *right_args, output_labels, optimize="greedy")
        B_mat = C.reshape(boundary_flat, full_rank)
        result = compress_from_factors_adaptive(
            A_mat,
            B_mat,
            tol=float(merge_tol),
            randomized=randomized,
            oversample=oversample,
            n_power_iter=n_power_iter,
            rng=rng,
        )
        factors = result.factors
        merge_info.used_rank = int(result.chosen_rank)
        merge_info.compressed = bool(result.chosen_rank < full_rank)
        merge_info.path = "explicit"
        merge_info.residual_ratio = float(result.residual_ratio)

    rank = factors.left.shape[-1]
    A_new = factors.left.reshape(*open_dims, rank) if open_dims else factors.left.reshape(rank)
    B_new = factors.right.reshape(*parent_boundary_dims, rank) if parent_boundary_dims else factors.right.reshape(rank)
    state = SeparatorState(
        open_labels=open_labels,
        open_dims=open_dims,
        boundary_labels=parent_boundary_labels,
        boundary_dims=parent_boundary_dims,
        A=A_new,
        B=B_new,
    )
    return state, merge_info
