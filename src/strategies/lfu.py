"""LFU eviction: evict leaves with smallest access frequency."""

from __future__ import annotations

from typing import Optional, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictOp, EvictionStrategy


class LFUStrategy(EvictionStrategy):
    def select_eviction(
        self, tree: RadixTree
    ) -> Optional[Tuple[RadixNode, EvictOp]]:
        leaves = tree.leaf_nodes()
        if not leaves:
            return None
        # Tie-break by creation order (older first) then last_access
        victim = min(leaves, key=lambda n: (n.access_count, n.creation_order, n.last_access))
        return (victim, "leaf")
