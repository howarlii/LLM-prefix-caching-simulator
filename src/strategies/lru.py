"""LRU eviction: evict leaves with smallest last-access time."""

from __future__ import annotations

from typing import List

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictionStrategy


class LRUStrategy(EvictionStrategy):
    def select_nodes(self, tree: RadixTree, num_nodes: int) -> List[RadixNode]:
        leaves = tree.leaf_nodes()
        if not leaves:
            return []
        leaves.sort(key=lambda n: n.last_access)
        return leaves[:num_nodes]
