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

from typing import List, Optional, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictOp, EvictionStrategy

# Minimum chain token length to place a mid-chain checkpoint.
_MIN_CHAIN_TOKENS_FOR_MID_CHECKPOINT = 2048


def _depth_tokens_from_checkpoint(node: RadixNode) -> int:
    """Token distance from the nearest ancestor with Mamba state (or root) to *node* (inclusive)."""
    d = len(node.page)
    cur = node.parent
    while cur is not None and cur.parent is not None:
        if cur.has_mamba_state:
            return d
        d += len(cur.page)
        cur = cur.parent
    return d


def _estimate_leaf_free(leaf: RadixNode) -> int:
    """Estimate tokens freed by removing a leaf and pruning its empty chain."""
    total = len(leaf.page)
    cur = leaf.parent
    while cur is not None and cur.parent is not None:
        if len(cur.children) > 1:
            break
        if cur.has_mamba_state:
            break
        total += len(cur.page)
        cur = cur.parent
    return total


class Marconi2Strategy(EvictionStrategy):
    """FLOP-aware eviction with checkpoint-relative cost and mid-chain checkpointing.

    Parameters
    ----------
    alpha:
        Weight for the FLOP-efficiency term in the eviction score.
    """

    def __init__(self, alpha: float = 1.5) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Unified scoring
    # ------------------------------------------------------------------

    def _collect_and_score(
        self, tree: RadixTree
    ) -> List[Tuple[float, RadixNode, EvictOp]]:
        """Score all eviction candidates on one axis.

        Returns list of ``(value_density, node, op)`` sorted ascending.
        """
        mte = tree.mamba_state_token_equiv

        mamba_candidates: List[RadixNode] = []
        leaf_candidates: List[RadixNode] = []
        for n in tree.iter_nodes():
            if n.is_leaf():
                leaf_candidates.append(n)
            if n.has_mamba_state and len(n.children) <= 1:
                mamba_candidates.append(n)

        all_nodes: List[Tuple[RadixNode, EvictOp]] = (
            [(n, "mamba") for n in mamba_candidates]
            + [(n, "leaf") for n in leaf_candidates]
        )
        if not all_nodes:
            return []

        recencies = [n.last_access for n, _ in all_nodes]
        depths = [_depth_tokens_from_checkpoint(n) for n, _ in all_nodes]

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

    def admit_mamba_state(self, node: RadixNode) -> bool:
        """Admit Mamba state only at turn-end nodes."""
        return node.is_turn_end

    def on_new_nodes_inserted(
        self, tree: RadixTree, new_nodes: List[RadixNode]
    ) -> None:
        """Handle fork-point parent admission + mid-chain checkpoint placement."""
        if not new_nodes:
            return

        # --- Fork-point parent ---
        parent = new_nodes[0].parent
        if parent is not None and parent is not tree.root:
            if len(parent.children) > 1 and not parent.has_mamba_state:
                tree.set_mamba_state(parent)

        # --- Mid-chain checkpoint at ~55% ---
        anchor = new_nodes[0].parent
        tokens_before_chain = 0
        cur = anchor
        while cur is not None and cur.parent is not None:
            if cur.has_mamba_state:
                break
            tokens_before_chain += len(cur.page)
            cur = cur.parent

        chain_tokens = sum(len(n.page) for n in new_nodes)
        total_span = tokens_before_chain + chain_tokens

        if total_span < _MIN_CHAIN_TOKENS_FOR_MID_CHECKPOINT:
            return

        target_pos = int(total_span * 0.55)

        if target_pos <= tokens_before_chain:
            remaining = tokens_before_chain - target_pos
            candidate = anchor
            while candidate is not None and candidate.parent is not None:
                if candidate.has_mamba_state:
                    break
                if remaining <= 0:
                    break
                remaining -= len(candidate.page)
                if remaining <= 0:
                    break
                candidate = candidate.parent
            if candidate is not None and candidate is not tree.root and not candidate.has_mamba_state:
                tree.set_mamba_state(candidate)
        else:
            offset_in_chain = target_pos - tokens_before_chain
            cumulative = 0
            for node in new_nodes:
                cumulative += len(node.page)
                if cumulative >= offset_in_chain:
                    if not node.has_mamba_state:
                        tree.set_mamba_state(node)
                    break

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

    def select_nodes(self, tree: RadixTree, num_nodes: int) -> List[RadixNode]:
        """Fallback for non-hybrid mode."""
        leaves = tree.leaf_nodes()
        if not leaves:
            return []
        recencies = [n.last_access for n in leaves]
        depths = [_depth_tokens_from_checkpoint(n) for n in leaves]
        min_r, max_r = min(recencies), max(recencies)
        min_d, max_d = min(depths), max(depths)

        scored: List[Tuple[float, RadixNode]] = []
        for n, r, d in zip(leaves, recencies, depths):
            norm_r = (r - min_r) / (max_r - min_r) if max_r > min_r else 0.0
            norm_d = (d - min_d) / (max_d - min_d) if max_d > min_d else 0.0
            scored.append((norm_r + self.alpha * norm_d, n))
        scored.sort(key=lambda x: x[0])
        return [n for _, n in scored[:num_nodes]]
