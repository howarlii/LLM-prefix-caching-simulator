"""LFU eviction: evict leaves with smallest access frequency."""

from __future__ import annotations

from typing import List

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictionStrategy


class LFUStrategy(EvictionStrategy):
    def select_nodes(self, tree: RadixTree, num_nodes: int) -> List[RadixNode]:
        leaves = tree.leaf_nodes()
        if not leaves:
            return []
        # Tie-break by creation order (older first) then last_access
        leaves.sort(key=lambda n: (n.access_count, n.creation_order, n.last_access))
        return leaves[:num_nodes]
