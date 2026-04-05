"""FIFO eviction: evict leaves with smallest creation order (first inserted)."""

from __future__ import annotations

from typing import Optional, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictOp, EvictionStrategy


class FIFOStrategy(EvictionStrategy):
    def select_eviction(
        self, tree: RadixTree
    ) -> Optional[Tuple[RadixNode, EvictOp]]:
        leaves = tree.leaf_nodes()
        if not leaves:
            return None
        victim = min(leaves, key=lambda n: (n.creation_order, n.last_access))
        return (victim, "leaf")
