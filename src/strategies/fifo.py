"""FIFO eviction: evict leaves with smallest creation order (first inserted)."""

from __future__ import annotations

from typing import List

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictionStrategy


class FIFOStrategy(EvictionStrategy):
    def select_nodes(self, tree: RadixTree, num_nodes: int) -> List[RadixNode]:
        leaves = tree.leaf_nodes()
        if not leaves:
            return []
        leaves.sort(key=lambda n: (n.creation_order, n.last_access))
        return leaves[:num_nodes]
