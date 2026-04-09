"""Marconi2: FLOP-aware eviction with checkpoint-relative cost + mid-chain checkpointing.

Differences from Marconi (v1)
-----------------------------
1. **Checkpoint-relative FLOPS**: The flop_efficiency term measures tokens from
   the nearest ancestor with a Mamba state (or root) to the current node,
   rather than always from root.  Evicting a Mamba checkpoint dynamically
   changes the effective FLOPS cost of downstream nodes.

2. **Unified eviction scoring** (same as Marconi v1): Both mamba-state
   demotions and leaf removals are scored on one axis via ``select_eviction``.

3. **Mid-chain Mamba state placement**: When a new leaf chain is inserted and
   its total span (from last checkpoint/root) >= 2048 tokens, a Mamba state
   is placed at roughly 55% of that span as an intermediate checkpoint.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from src.radix_tree import PageKey, RadixNode, RadixTree
from src.strategies.base import (
    EvictOp,
    EvictionStrategy,
    PageStatus,
    RequestPlan,
)

# Minimum chain token length to place a mid-chain checkpoint.
_MIN_CHAIN_TOKENS_FOR_MID_CHECKPOINT = 2048



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


class Marconi2Strategy(EvictionStrategy):
    """FLOP-aware eviction with checkpoint-relative cost and mid-chain checkpointing.

    Parameters
    ----------
    alpha:
        Weight for the FLOP-efficiency term in the eviction score.
    use_checkpoint_relative_evict:
        If True (default), measure FLOP depth from nearest mamba checkpoint
        rather than from root.  Setting False reverts to Marconi-v1 scoring.
    use_mid_chain_checkpoint:
        If True (default), place a mid-chain mamba state at ~55% of long
        new chains.  Setting False disables this behaviour.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        use_checkpoint_relative_evict: bool = True,
        use_mid_chain_checkpoint: bool = True,
    ) -> None:
        self.alpha = alpha
        self.use_checkpoint_relative_evict = use_checkpoint_relative_evict
        self.use_mid_chain_checkpoint = use_mid_chain_checkpoint

    # ------------------------------------------------------------------
    # Unified scoring
    # ------------------------------------------------------------------

    def _collect_and_score(
        self, tree: RadixTree
    ) -> List[Tuple[float, RadixNode, EvictOp]]:
        """Score all eviction candidates on one axis.

        Uses top-down BFS to compute depth_tokens and checkpoint-relative
        depth inline, avoiding O(depth) parent walks per node.

        Returns list of ``(value_density, node, op)`` sorted ascending.
        """
        mte = tree.mamba_state_token_equiv
        use_cp = self.use_checkpoint_relative_evict

        # Top-down BFS: depth_tokens is precomputed on each node; only
        # checkpoint-relative depth needs tracking.
        # node_depths maps node id -> (depth_tokens, depth_from_checkpoint)
        node_depths: dict[int, Tuple[int, int]] = {}
        q: deque[Tuple[RadixNode, int]] = deque()
        for child in tree.root.children.values():
            q.append((child, child.num_tokens))

        while q:
            node, depth_from_cp = q.popleft()
            node_depths[id(node)] = (node.depth_tokens, depth_from_cp)

            for ch in node.children.values():
                if node.has_mamba_state:
                    ch_from_cp = ch.num_tokens
                else:
                    ch_from_cp = depth_from_cp + ch.num_tokens
                q.append((ch, ch_from_cp))

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
        depths = [
            node_depths[id(n)][1] if use_cp else node_depths[id(n)][0]
            for n, _ in all_nodes
        ]

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

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def plan_request(
        self,
        tree: RadixTree,
        matched_nodes: List[RadixNode],
        remaining_pages: List[PageKey],
    ) -> RequestPlan:
        """Admit every remaining page; place mamba at turn-end,
        fork-point parent, and (optionally) at the mid-chain ~55% point
        of long new chains."""
        if not remaining_pages:
            return RequestPlan(remaining=[])

        statuses: List[PageStatus] = [PageStatus.KV_ONLY] * len(remaining_pages)
        fork_point = False

        if tree.mamba_state_token_equiv == 0:
            return RequestPlan(remaining=statuses)

        parent = matched_nodes[-1] if matched_nodes else tree.root
        parent_depth = parent.depth_tokens

        # --- Turn-end mamba ---
        statuses[-1] = PageStatus.KV_AND_MAMBA
        end_depth = parent_depth + sum(len(p) for p in remaining_pages)

        # --- Fork-point parent ---
        if (
            parent is not tree.root
            and len(parent.children) >= 1
            and not parent.has_mamba_state
        ):
            fork_point = True

        # --- Mid-chain checkpoint at ~55% of (last_checkpoint → end) ---
        if not self.use_mid_chain_checkpoint:
            return RequestPlan(
                remaining=statuses, mamba_on_matched_parent=fork_point
            )

        # Walk up matched_nodes to find the last mamba checkpoint.
        last_checkpoint_depth = 0
        for n in reversed(matched_nodes):
            if n.has_mamba_state:
                last_checkpoint_depth = n.depth_tokens
                break

        total_span = end_depth - last_checkpoint_depth
        if total_span < _MIN_CHAIN_TOKENS_FOR_MID_CHECKPOINT:
            return RequestPlan(
                remaining=statuses, mamba_on_matched_parent=fork_point
            )

        target_depth = last_checkpoint_depth + int(total_span * 0.55)
        # Convert target_depth → page index within remaining_pages.
        if target_depth > parent_depth and target_depth < end_depth:
            offset = target_depth - parent_depth
            cum = 0
            for i, p in enumerate(remaining_pages):
                cum += len(p)
                if cum >= offset:
                    # Place mamba at the end of page i (snapped to the
                    # smallest page boundary at-or-past target).
                    if statuses[i] != PageStatus.KV_AND_MAMBA:
                        statuses[i] = PageStatus.KV_AND_MAMBA
                    break

        return RequestPlan(
            remaining=statuses, mamba_on_matched_parent=fork_point
        )

    # ------------------------------------------------------------------
    # Eviction — unified via select_eviction
    # ------------------------------------------------------------------

    def select_eviction(
        self, tree: RadixTree
    ) -> Optional[Tuple[RadixNode, EvictOp]]:
        """Pick the single best eviction action via unified scoring."""
        scored = self._collect_and_score(tree)
        if not scored:
            return None
        _, node, op = scored[0]
        return (node, op)
