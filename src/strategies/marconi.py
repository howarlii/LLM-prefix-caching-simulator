"""Marconi-style FLOP-aware eviction + selective Mamba-state admission.

Admission (hybrid mode only)
----------------------------
Mamba SSM states are admitted at two types of high-value positions:

1. **Turn-end nodes** – the last inserted node of every request
   (``node.is_turn_end``).

2. **Fork-point parents** – when a newly inserted suffix creates a
   second branch off an existing tree node, that node becomes a fork
   (shared prefix split).

Eviction (FLOP-aware, unified)
------------------------------
All eviction candidates — both Mamba-state demotions and full leaf
removals — are scored on one axis:

    value_density = raw_score / capacity_freed

where ``raw_score = recency + alpha * flop_efficiency`` (both normalised
to [0, 1]).  The candidate with the **lowest** value density is evicted.

* ``recency(n)`` – normalised ``last_access`` timestamp.
* ``flop_efficiency(n)`` – normalised cumulative token depth from root
  to *n* (higher = longer sequence = more FLOPs saved).

Candidate sets:
* Mamba-state demotion: ``has_mamba_state and len(children) <= 1``.
  Frees ``mamba_state_token_equiv`` capacity.
* Leaf removal: leaf nodes.  Frees page tokens (+ pruned chain).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictOp, EvictionStrategy


def _depth_tokens(node: RadixNode) -> int:
    """Total number of tokens on the path from root down to *node* (inclusive)."""
    d = 0
    n: RadixNode | None = node
    while n is not None and n.parent is not None:
        d += n.num_tokens
        n = n.parent
    return d


def _estimate_leaf_free(leaf: RadixNode) -> int:
    """Estimate tokens freed by removing a leaf and pruning its empty chain."""
    total = leaf.num_tokens
    cur = leaf.parent
    while cur is not None and cur.parent is not None:
        if len(cur.children) > 1:
            break
        if cur.has_mamba_state:
            break
        total += cur.num_tokens
        cur = cur.parent
    return total


class MarconiStrategy(EvictionStrategy):
    """FLOP-aware eviction with selective Mamba-state admission (Marconi paper).

    Parameters
    ----------
    alpha:
        Weight for the FLOP-efficiency term in the eviction score.
        Higher values favour retaining long-sequence Mamba states.
    """

    def __init__(self, alpha: float = 1.5) -> None:
        self.alpha = alpha

    @property
    def drop_partial_last_page(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def admit_mamba_state(self, node: RadixNode) -> bool:
        """Admit Mamba state only at turn-end nodes (last token of a request)."""
        return node.is_turn_end

    def on_new_nodes_inserted(
        self, tree: RadixTree, new_nodes: List[RadixNode]
    ) -> None:
        """Admit Mamba state at fork-point parents created by this insertion."""
        if not new_nodes:
            return
        parent = new_nodes[0].parent
        if parent is None or parent is tree.root:
            return
        if len(parent.children) > 1 and not parent.has_mamba_state:
            tree.set_mamba_state(parent)

    # ------------------------------------------------------------------
    # Unified eviction
    # ------------------------------------------------------------------

    def _collect_and_score(
        self, tree: RadixTree
    ) -> List[Tuple[float, RadixNode, EvictOp]]:
        """Score all eviction candidates on one axis.

        Returns list of ``(value_density, node, op)`` sorted ascending.
        """
        mte = tree.mamba_state_token_equiv

        leaf_candidates: List[RadixNode] = list(tree.leaf_node_set())
        mamba_candidates: List[RadixNode] = [
            n for n in tree.mamba_state_node_set() if len(n.children) <= 1
        ]

        all_nodes: List[Tuple[RadixNode, EvictOp]] = (
            [(n, "mamba") for n in mamba_candidates]
            + [(n, "leaf") for n in leaf_candidates]
        )
        if not all_nodes:
            return []

        recencies = [n.last_access for n, _ in all_nodes]
        depths = [_depth_tokens(n) for n, _ in all_nodes]

        min_r, max_r = min(recencies), max(recencies)
        min_d, max_d = min(depths), max(depths)

        scored: List[Tuple[float, RadixNode, EvictOp]] = []
        for (n, op), r, d in zip(all_nodes, recencies, depths):
            norm_r = (r - min_r) / (max_r - min_r) if max_r > min_r else 0.0
            norm_d = (d - min_d) / (max_d - min_d) if max_d > min_d else 0.0
            raw_score = norm_r + self.alpha * norm_d

            if op == "mamba":
                freed = max(mte, 1)
            else:
                freed = max(_estimate_leaf_free(n), 1)
                if n.has_mamba_state:
                    freed += mte

            scored.append((raw_score / freed, n, op))

        scored.sort(key=lambda x: x[0])
        return scored

    def select_eviction(
        self, tree: RadixTree
    ) -> Optional[Tuple[RadixNode, EvictOp]]:
        """Pick the single best eviction action via unified scoring."""
        scored = self._collect_and_score(tree)
        if not scored:
            return None
        _, node, op = scored[0]
        return (node, op)
