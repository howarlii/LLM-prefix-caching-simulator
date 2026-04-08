"""Branch strategy: Marconi-style admission + plain LRU eviction.

Admission (hybrid mode only)
----------------------------
Mirrors the original Marconi paper's admission policy:

1. **Turn-end nodes** — the last inserted node of every request
   (``node.is_turn_end``).
2. **Fork-point parents** — when a newly inserted suffix branches off an
   existing tree node, that parent gets a Mamba state stored.

No mid-chain checkpointing — only request boundaries and structural
fork points are tagged.

Eviction
--------
Plain LRU over all evictable candidates. Two candidate types (matching
``marconi3_ev1`` granularity):

* **Leaf nodes** (``len(children) == 0``) → removed entirely.
* **Internal nodes carrying a Mamba state** → only the Mamba state is
  dropped (the KV cache and downstream subtree stay alive).

The candidate with the smallest "effective last access" is evicted.

Touch policy
------------
By default ("branch") the strategy follows the radix tree's normal touch
behaviour: every matched node on a cache hit refreshes its access time.

When ``newtouch=True`` ("branch_nt") only a subset of matched nodes is
refreshed:

* The deepest matched node (last hit) is always refreshed.
* For each ancestor matched node ``n``, look at the *previous checkpoint*
  — the deepest mamba-state node at-or-below ``n`` on the matched path
  (i.e. the most recently visited checkpoint when walking from the
  deepest hit upward toward ``n``):

  - if that checkpoint is currently a **leaf** (no children) → do not
    refresh ``n``;
  - if it is currently a **branch point** (multiple children) → refresh
    ``n``;
  - otherwise (no checkpoint below ``n``, or a single-child checkpoint)
    → do not refresh ``n``.

To implement this without fighting the radix tree (which always touches
matched nodes inside ``simulate_request`` before any strategy hook
runs), the strategy maintains a parallel ``_branch_lru`` attribute on
each managed node and sorts evictions by it.  ``_branch_lru`` is
initialised whenever a node is first inserted via the strategy hook,
and only updated by ``on_cache_hit`` for nodes the policy chooses to
refresh.  Split-induced suffixes (which have no strategy hook) fall
back to the radix tree's ``last_access`` field.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictOp, EvictionStrategy


def _effective_ts(node: RadixNode) -> int:
    """LRU sort key: prefer the strategy-managed timestamp, fall back to last_access."""
    return getattr(node, "_branch_lru", node.last_access)


class BranchStrategy(EvictionStrategy):
    """Marconi-style admission with LRU eviction at marconi3_ev1 granularity.

    Parameters
    ----------
    newtouch:
        If ``False`` (default, "branch"), every matched node is refreshed
        on a cache hit, matching standard LRU semantics.
        If ``True`` ("branch_nt"), only the deepest matched node and
        selected ancestor checkpoints are refreshed (see module docstring
        for the precise rule).
    """

    def __init__(self, newtouch: bool = False) -> None:
        self.newtouch = newtouch

    @property
    def drop_partial_last_page(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Admission (identical to MarconiStrategy)
    # ------------------------------------------------------------------

    def admit_mamba_state(self, node: RadixNode) -> bool:
        """Admit Mamba state only at turn-end nodes (last token of a request)."""
        return node.is_turn_end

    def on_new_nodes_inserted(
        self, tree: RadixTree, new_nodes: List[RadixNode]
    ) -> None:
        """Initialise per-node LRU clock and admit fork-point mamba state."""
        if not new_nodes:
            return
        ts = tree.clock
        for n in new_nodes:
            n._branch_lru = ts  # type: ignore[attr-defined]

        parent = new_nodes[0].parent
        if parent is None or parent is tree.root:
            return
        if len(parent.children) > 1 and not parent.has_mamba_state:
            tree.set_mamba_state(parent)

    # ------------------------------------------------------------------
    # Touch policy
    # ------------------------------------------------------------------

    def on_cache_hit(
        self, tree: RadixTree, matched_nodes: List[RadixNode]
    ) -> None:
        if not matched_nodes:
            return
        ts = tree.clock

        if not self.newtouch:
            # Default: refresh all matched nodes (standard LRU).
            for n in matched_nodes:
                n._branch_lru = ts  # type: ignore[attr-defined]
            return

        # newtouch: deepest hit + selective ancestors.
        last = matched_nodes[-1]
        last._branch_lru = ts  # type: ignore[attr-defined]

        # Walk from deepest matched upward.  ``cur_cp`` tracks the deepest
        # mamba-state node we have already passed in this walk — i.e. the
        # "previous checkpoint" relative to the node we are about to look
        # at next.  When considering the next ancestor n, ``cur_cp`` is the
        # deepest checkpoint strictly below n on the matched path.
        cur_cp: Optional[RadixNode] = last if last.has_mamba_state else None
        for i in range(len(matched_nodes) - 2, -1, -1):
            n = matched_nodes[i]
            if cur_cp is not None:
                num_ch = len(cur_cp.children)
                if num_ch > 1:
                    # Previous checkpoint is a branch point → refresh n.
                    n._branch_lru = ts  # type: ignore[attr-defined]
                # num_ch == 0 (leaf) or num_ch == 1 (single-child): skip.
            # Promote n to the current checkpoint for nodes higher up.
            if n.has_mamba_state:
                cur_cp = n

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _collect_candidates(
        self, tree: RadixTree
    ) -> List[Tuple[RadixNode, EvictOp]]:
        """Enumerate eviction candidates in the marconi3_ev1 style.

        * Leaf nodes → ``"leaf"`` (full removal).
        * Internal nodes with a Mamba state → ``"mamba"`` (state-only drop).
        """
        candidates: List[Tuple[RadixNode, EvictOp]] = []
        q: deque[RadixNode] = deque(tree.root.children.values())
        while q:
            node = q.popleft()
            if not node.children:
                candidates.append((node, "leaf"))
            elif node.has_mamba_state:
                candidates.append((node, "mamba"))
            q.extend(node.children.values())
        return candidates

    def select_eviction(
        self, tree: RadixTree
    ) -> Optional[Tuple[RadixNode, EvictOp]]:
        candidates = self._collect_candidates(tree)
        if not candidates:
            return None
        return min(candidates, key=lambda c: _effective_ts(c[0]))
