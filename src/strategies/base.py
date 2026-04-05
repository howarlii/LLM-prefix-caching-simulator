"""Abstract eviction / admission strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from src.radix_tree import RadixNode, RadixTree

EvictOp = Literal["mamba", "leaf"]


class EvictionStrategy(ABC):
    """Decides which cached pages (tree nodes) to evict and which to admit.

    Eviction
    --------
    ``select_eviction``
        The main entry point called by the simulator when over capacity.
        Returns ``(node, "mamba")`` to drop only the Mamba state, or
        ``(node, "leaf")`` to remove the leaf entirely, or ``None`` if
        nothing can be evicted.

        The default implementation falls back to
        ``select_mamba_state_evictions`` then ``select_nodes`` (preserving
        legacy behaviour for simple strategies like LRU/LFU/FIFO).
        Strategies that want unified scoring (e.g. Marconi) should override
        ``select_eviction`` directly.

    Hybrid-model hooks
    ------------------
    ``admit_mamba_state``
        Called for each newly inserted node; return ``True`` to store a Mamba
        state at that node.  Default: always admit.
    """

    @abstractmethod
    def select_nodes(self, tree: RadixTree, num_nodes: int) -> List[RadixNode]:
        """Return up to ``num_nodes`` leaf nodes to remove entirely.

        Implementations should only return nodes that are safe to delete
        (typically leaves). Fewer than ``num_nodes`` may be returned if the
        tree has fewer evictable nodes.
        """
        raise NotImplementedError

    def admit_mamba_state(self, node: RadixNode) -> bool:
        """Return ``True`` if a Mamba state should be stored for *node*.

        Called once per newly inserted node, only when the simulator is
        running in hybrid mode (``mamba_state_token_equiv > 0``).
        Default: always admit.
        """
        return True

    def select_mamba_state_evictions(
        self, tree: RadixTree, num_states: int
    ) -> List[RadixNode]:
        """Return up to ``num_states`` nodes from which to evict *only* the
        Mamba state (the KV cache pages are kept in place).

        Default: return empty list (never demote).
        """
        return []

    def select_eviction(
        self, tree: RadixTree
    ) -> Optional[Tuple[RadixNode, EvictOp]]:
        """Pick the single best eviction action.

        Returns ``(node, "mamba")`` to drop only the Mamba state, or
        ``(node, "leaf")`` to remove the leaf entirely, or ``None``.

        Default: try ``select_mamba_state_evictions`` first, then
        ``select_nodes`` (legacy two-phase behaviour for simple strategies).
        Override this for unified scoring.
        """
        mamba = self.select_mamba_state_evictions(tree, 1)
        if mamba:
            return (mamba[0], "mamba")
        leaves = self.select_nodes(tree, 1)
        if leaves:
            return (leaves[0], "leaf")
        return None

    def on_cache_hit(
        self, tree: RadixTree, matched_nodes: List[RadixNode]
    ) -> None:
        """Called once per request with the nodes that were matched (cache hit).

        Strategies that maintain per-node metadata (e.g. CRF scores) can
        override this to update their bookkeeping on hits.
        Default: no-op.
        """
        return

    def on_new_nodes_inserted(
        self, tree: RadixTree, new_nodes: List[RadixNode]
    ) -> None:
        """Called once per request after all new suffix nodes are inserted and
        ``admit_mamba_state`` has been applied to each.

        Strategies that need to inspect the tree structure *after* a full
        insertion (e.g. to detect newly-created fork points) can override this.
        Default: no-op.
        """
        return
