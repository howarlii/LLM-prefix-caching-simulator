"""Abstract eviction / admission strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.radix_tree import RadixNode, RadixTree


class EvictionStrategy(ABC):
    """Decides which cached pages (tree nodes) to evict and which to admit.

    Hybrid-model hooks
    ------------------
    ``admit_mamba_state``
        Called for each newly inserted node; return ``True`` to store a Mamba
        state at that node.  Default: always admit (store Mamba state for
        every node).
    ``select_mamba_state_evictions``
        Return nodes whose *Mamba state only* should be evicted (KV cache
        kept).  The simulator tries these before full-node evictions.
        Default: never evict Mamba states separately.
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

        The simulator calls this *before* ``select_nodes`` when capacity is
        exceeded, so strategies can choose to demote nodes (KV-only) rather
        than dropping them entirely.  Default: return empty list (never demote).
        """
        return []

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
