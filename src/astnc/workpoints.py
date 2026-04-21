from __future__ import annotations

from astnc.errors import InvalidWorkpointError
from astnc.types import Workpoint


_WORKPOINTS: dict[str, Workpoint] = {
    "l1": Workpoint(
        name="l1",
        description="Most accurate default ASTNC setting.",
        method_options={
            "leaf_tol": 5e-4,
            "merge_tol": 2e-3,
        },
    ),
    "l2": Workpoint(
        name="l2",
        description="Balanced accuracy and runtime.",
        method_options={
            "leaf_tol": 1e-3,
            "merge_tol": 5e-3,
        },
    ),
    "l3": Workpoint(
        name="l3",
        description="Fastest default ASTNC setting with looser tolerances.",
        method_options={
            "leaf_tol": 3e-3,
            "merge_tol": 1.5e-2,
        },
    ),
}


def available_workpoints() -> tuple[str, ...]:
    return tuple(_WORKPOINTS.keys())


def get_workpoint(name: str) -> Workpoint:
    key = str(name).lower()
    try:
        return _WORKPOINTS[key]
    except KeyError as exc:
        raise InvalidWorkpointError(f"Unknown workpoint: {name!r}") from exc
