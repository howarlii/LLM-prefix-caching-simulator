"""Marconi-style FLOP-aware eviction + selective Mamba-state admission.

Admission (hybrid mode only)
----------------------------
Mamba SSM states are admitted at two types of high-value positions:

1. **Turn-end nodes** – the last inserted node of every request
   (``node.is_turn_end``).  Multi-turn continuations are the primary
   reuse pattern, so the SSM state at the end of each response is
   always worth storing.

2. **Fork-point parents** – when a newly inserted suffix creates a
   second branch off an existing tree node, that node becomes a fork
   (shared prefix split).  Its Mamba state is admitted via
   ``on_new_nodes_inserted`` so it is available for future requests
   that share the same prefix.

Eviction (FLOP-aware)
---------------------
Score for node *n*:

    S(n) = recency(n) + α · flop_efficiency(n)

Both terms normalised to [0, 1] over the current candidate set.

* ``recency(n)`` – normalised ``last_access`` timestamp (higher = more
  recently used).
* ``flop_efficiency(n)`` – normalised cumulative token depth from root
  to *n* (higher = longer sequence = more FLOPs saved per byte of
  Mamba state).

Nodes with the **lowest** score are evicted first, which prefers
evicting short/stale entries and retaining long/recent ones.

Evictable node sets
~~~~~~~~~~~~~~~~~~~
* ``select_mamba_state_evictions``: nodes with ``has_mamba_state`` and
  ``len(children) <= 1``.  Dropping only the Mamba state from a
  single-child intermediate node frees the fixed SSM-state budget
  while keeping its KV pages (absorbed by its child's path).
* ``select_nodes``: leaf nodes (standard full removal).
"""

from __future__ import annotations

from typing import List, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictionStrategy


def _depth_tokens(node: RadixNode) -> int:
    """Total number of tokens on the path from root down to *node* (inclusive)."""
    d = 0
    n: RadixNode | None = node
    while n is not None and n.parent is not None:
        d += len(n.page)
        n = n.parent
    return d


def _flop_aware_sort(
    nodes: List[RadixNode], alpha: float
) -> List[Tuple[float, RadixNode]]:
    """Return ``(score, node)`` pairs sorted ascending (lowest score first).

    Score = recency + alpha * flop_efficiency, both in [0, 1].
    Ties broken by depth (shallower = lower score = evict sooner).
    """
    if not nodes:
        return []

    recencies = [n.last_access for n in nodes]
    depths = [_depth_tokens(n) for n in nodes]

    min_r, max_r = min(recencies), max(recencies)
    min_d, max_d = min(depths), max(depths)

    scored: List[Tuple[float, RadixNode]] = []
    for n, r, d in zip(nodes, recencies, depths):
        norm_r = (r - min_r) / (max_r - min_r) if max_r > min_r else 0.0
        norm_d = (d - min_d) / (max_d - min_d) if max_d > min_d else 0.0
        scored.append((norm_r + alpha * norm_d, n))

    scored.sort(key=lambda x: x[0])
    return scored


class MarconiStrategy(EvictionStrategy):
    """FLOP-aware eviction with selective Mamba-state admission (Marconi paper).

    Parameters
    ----------
    alpha:
        Weight for the FLOP-efficiency term in the eviction score.
        Higher values favour retaining long-sequence Mamba states.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def admit_mamba_state(self, node: RadixNode) -> bool:
        """Admit Mamba state only at turn-end nodes (last token of a request)."""
        return node.is_turn_end

    def on_new_nodes_inserted(
        self, tree: RadixTree, new_nodes: List[RadixNode]
    ) -> None:
        """Admit Mamba state at fork-point parents created by this insertion.

        When the first new node creates a second branch off an existing tree
        node (len(parent.children) > 1), that parent has just become a shared
        prefix split point.  Its Mamba state is admitted here so future
        requests reaching that fork can skip recomputation.
        """
        if not new_nodes:
            return
        parent = new_nodes[0].parent
        if parent is None or parent is tree.root:
            return
        # A fork point: parent now has more than one child branch.
        if len(parent.children) > 1 and not parent.has_mamba_state:
            tree.set_mamba_state(parent)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def select_mamba_state_evictions(
        self, tree: RadixTree, num_states: int
    ) -> List[RadixNode]:
        """Evict Mamba states from low-value nodes with <= 1 child.

        Single-child intermediate nodes represent non-shared path segments;
        their SSM states can be dropped while their KV pages remain accessible
        through their child's path.
        """
        candidates = [
            n
            for n in tree.iter_nodes()
            if n.has_mamba_state and len(n.children) <= 1
        ]
        scored = _flop_aware_sort(candidates, self.alpha)
        return [n for _, n in scored[:num_states]]

    def select_nodes(self, tree: RadixTree, num_nodes: int) -> List[RadixNode]:
        """Evict leaf nodes using FLOP-aware scoring."""
        leaves = tree.leaf_nodes()
        if not leaves:
            return []
        scored = _flop_aware_sort(leaves, self.alpha)
        return [n for _, n in scored[:num_nodes]]
