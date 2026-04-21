from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from astnc.internal.cache import SeparatorStateCache
from astnc.internal.network import TensorNetwork, TensorNode

@dataclass
class ContractionCache:
    """Reusable cache container for separator states."""

    state_cache: SeparatorStateCache = field(default_factory=SeparatorStateCache)

    def summary(self) -> Dict[str, float | int | bool]:
        return self.state_cache.summary()


TensorNetworkLike = TensorNetwork
TensorNodeLike = TensorNode
Options = Mapping[str, Any]
