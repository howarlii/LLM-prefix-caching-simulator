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

from src.model_config import DEFAULT_MODEL, ModelConfig
from src.radix_tree import PageKey, RadixNode, RadixTree
from src.strategies.base import (
    EvictOp,
    EvictionStrategy,
    PageStatus,
    RequestPlan,
    compute_min_mamba_admit_depth,
)


def _effective_ts(node: RadixNode) -> int:
    """LRU sort key: prefer the strategy-managed timestamp, fall back to last_access."""
    return getattr(node, "_branch_lru", node.last_access)


def _ef_evictable(node: RadixNode, root: RadixNode) -> bool:
    """``_ef`` filter: evictable iff ``len(children)==1`` OR no other node
    with a mamba state lies strictly between ``node`` and the previous
    branching point (or root) along the ancestor chain."""
    if len(node.children) == 1:
        return True
    cur = node.parent
    while cur is not None and cur is not root:
        if len(cur.children) >= 2:
            break
        if cur.has_mamba_state:
            return False
        cur = cur.parent
    return True


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
    model:
        Architecture description; used (with ``gpu_flops`` /
        ``pcie_bandwidth``) to compute a hardware-aware depth threshold
        below which admitting a Mamba state would cost more (PCIe load
        time) than recomputing the prefix from scratch.
    gpu_flops / pcie_bandwidth:
        Hardware throughput for the depth-threshold check.  When either
        is missing the filter is disabled and the strategy admits states
        regardless of depth (matching the original behaviour).
    """

    def __init__(
        self,
        newtouch: bool = False,
        evict_filter: bool = False,
        model: ModelConfig = DEFAULT_MODEL,
        gpu_flops: Optional[float] = None,
        pcie_bandwidth: Optional[float] = None,
    ) -> None:
        self.newtouch = newtouch
        self.evict_filter = evict_filter
        self.model = model
        self.gpu_flops = gpu_flops
        self.pcie_bandwidth = pcie_bandwidth
        self._min_mamba_admit_depth = compute_min_mamba_admit_depth(
            model, gpu_flops, pcie_bandwidth
        )

    # ------------------------------------------------------------------
    # Admission (identical to MarconiStrategy)
    # ------------------------------------------------------------------

    def plan_request(
        self,
        tree: RadixTree,
        matched_nodes: List[RadixNode],
        remaining_pages: List[PageKey],
    ) -> RequestPlan:
        """Admit every remaining page; place mamba at turn-end + fork-
        point parent when their depths clear the PCIe break-even threshold."""
        if not remaining_pages:
            return RequestPlan(remaining=[])

        statuses: List[PageStatus] = [PageStatus.KV_ONLY] * len(remaining_pages)
        fork_point = False

        if tree.mamba_state_token_equiv == 0:
            return RequestPlan(remaining=statuses)

        parent = matched_nodes[-1] if matched_nodes else tree.root
        parent_depth = parent.depth_tokens
        end_depth = parent_depth + sum(len(p) for p in remaining_pages)
        threshold = self._min_mamba_admit_depth

        if threshold == 0 or end_depth >= threshold:
            statuses[-1] = PageStatus.KV_AND_MAMBA

        if (
            parent is not tree.root
            and len(parent.children) >= 1
            and not parent.has_mamba_state
            and (threshold == 0 or parent_depth >= threshold)
        ):
            fork_point = True

        return RequestPlan(
            remaining=statuses, mamba_on_matched_parent=fork_point
        )

    def on_nodes_inserted(
        self, tree: RadixTree, new_nodes: List[RadixNode]
    ) -> None:
        """Initialise the per-node ``_branch_lru`` timestamp."""
        if not new_nodes:
            return
        ts = tree.clock
        for n in new_nodes:
            n._branch_lru = ts  # type: ignore[attr-defined]

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

        for i in range(len(matched_nodes) - 1, -1, -1):
            n = matched_nodes[i]
            if n.has_mamba_state:
                n._branch_lru = ts  # type: ignore[attr-defined]
                break

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
        if self.evict_filter:
            root = tree.root
            candidates = [c for c in candidates if _ef_evictable(c[0], root)]
        return candidates

    def select_eviction(
        self, tree: RadixTree
    ) -> Optional[Tuple[RadixNode, EvictOp]]:
        candidates = self._collect_candidates(tree)
        if not candidates:
            return None
        return min(candidates, key=lambda c: _effective_ts(c[0]))
