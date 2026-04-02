"""Abstract eviction strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.radix_tree import RadixNode, RadixTree


class EvictionStrategy(ABC):
    """Decides which cached pages (tree nodes) to evict when over capacity."""

    @abstractmethod
    def select_nodes(self, tree: RadixTree, num_nodes: int) -> List[RadixNode]:
        """Return up to ``num_nodes`` leaf nodes to remove.

        Implementations should only return nodes that are safe to delete
        (typically leaves). Fewer than ``num_nodes`` may be returned if the
        tree has fewer evictable nodes.
        """
        raise NotImplementedError
