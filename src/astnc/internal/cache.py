from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Tuple

if TYPE_CHECKING:
    from astnc.internal.state import SeparatorState


LocalBlockKey = Tuple[Tuple[int, Tuple[int, ...]], ...]
StateCacheKey = Tuple[Tuple[int, ...], LocalBlockKey, Tuple[Tuple[str, object], ...]]


def make_local_block_key(open_labels: Iterable[int], slice_map: dict[int, Iterable[int]]) -> LocalBlockKey:
    entries = []
    for label in open_labels:
        if label in slice_map:
            entries.append((int(label), tuple(int(idx) for idx in slice_map[label])))
    return tuple(entries)


class SeparatorStateCache:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._store: Dict[StateCacheKey, "SeparatorState"] = {}
        self.requests = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: StateCacheKey) -> "SeparatorState | None":
        if not self.enabled:
            return None
        self.requests += 1
        cached = self._store.get(key)
        if cached is None:
            self.misses += 1
            return None
        self.hits += 1
        return cached

    def put(self, key: StateCacheKey, state: "SeparatorState") -> None:
        if not self.enabled:
            return
        self._store[key] = state

    def summary(self) -> dict[str, float | int | bool]:
        requests = max(1, self.requests)
        return {
            "cache_enabled": bool(self.enabled),
            "cache_requests": int(self.requests),
            "cache_hits": int(self.hits),
            "cache_misses": int(self.misses),
            "cache_hit_rate": float(self.hits / requests) if self.enabled else 0.0,
            "num_cached_states": int(len(self._store)),
        }

