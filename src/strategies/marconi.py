"""Marconi-style FLOP-aware eviction + selective Mamba-state admission.

Admission (hybrid mode only)
----------------------------
Mamba SSM states are admitted at two types of high-value positions:

1. **Turn-end nodes** – the last inserted node of every request
   (``node.is_turn_end``).

2. **Fork-point parents** – when a newly inserted suffix creates a
   second branch off an existing tree node, that node becomes a fork
   (shared prefix split).

Eviction (FLOP-aware, aligned with original Marconi paper)
----------------------------------------------------------
All nodes with ``len(children) <= 1`` (excluding root) are candidates.
Each is scored on one axis:

    score = alpha * norm_flop_eff + norm_recency

The candidate with the **lowest** score is evicted.

* ``recency(n)`` – ``1 / (current_ts - last_access)``, normalised to [0, 1].
* ``flop_efficiency(n)`` – incremental FLOPs saved (relative to parent)
  divided by total memory cost of the path, normalised to [0, 1].
  Matches the original ``radix_cache_hybrid.evict_v2`` formula including
  the MLP bug (uses ``get_attn_flop`` instead of ``get_mlp_flop`` for
  the parent subtraction term).

Candidate operations:
* Leaf nodes (``len(children) == 0``): removed entirely.
* Single-child nodes with Mamba state: Mamba state demoted.
* Single-child nodes without Mamba state: treated as leaf removal.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from src.model_config import (
    DEFAULT_MODEL,
    ModelConfig,
    _attn_flop,
    _kvs_size,
    _mamba1_flop,
    _mamba_state_size,
    _mlp_flop,
)
from src.radix_tree import PageKey, RadixNode, RadixTree
from src.strategies.base import (
    EvictOp,
    EvictionStrategy,
    PageStatus,
    RequestPlan,
)


def _normalize(values: List[float], default: float = 1.0) -> List[float]:
    """Min-max normalisation matching the original Marconi ``_normalize``."""
    if len(values) > 1:
        min_val = min(values)
        max_val = max(values)
        if min_val != max_val:
            return [(v - min_val) / (max_val - min_val) for v in values]
    return [default] * len(values)


class MarconiStrategy(EvictionStrategy):
    """FLOP-aware eviction with selective Mamba-state admission (Marconi paper).

    Parameters
    ----------
    alpha:
        Weight for the FLOP-efficiency term in the eviction score.
        Higher values favour retaining long-sequence Mamba states.
    model:
        Model architecture configuration.  Provides layer counts, hidden
        dimension, and SSM state dimension for FLOP computation.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        model: ModelConfig = DEFAULT_MODEL,
    ) -> None:
        self.alpha = alpha
        self.model = model
        self.num_ssm_layers = model.num_ssm_layers
        self.num_attn_layers = model.num_attn_layers
        self.num_mlp_layers = model.num_mlp_layers
        self.d = model.d_model
        self.n = model.ssm_state_dim

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def plan_request(
        self,
        tree: RadixTree,
        matched_nodes: List[RadixNode],
        remaining_pages: List[PageKey],
    ) -> RequestPlan:
        """Admit every remaining page.  In hybrid mode, put mamba state
        at the turn-end page and (if this insertion creates a branching
        point) on the deepest matched node (the fork-point parent)."""
        if not remaining_pages:
            return RequestPlan(remaining=[])

        default_status = PageStatus.KV_ONLY
        statuses = [default_status] * len(remaining_pages)
        fork_point = False

        if tree.mamba_state_token_equiv > 0:
            # Turn-end mamba.
            statuses[-1] = PageStatus.KV_AND_MAMBA

            # Fork-point parent: if the deepest matched node already has
            # at least one child, this new insertion turns it into a
            # branching point.
            if matched_nodes:
                parent = matched_nodes[-1]
                if (
                    parent is not tree.root
                    and len(parent.children) >= 1
                    and not parent.has_mamba_state
                ):
                    fork_point = True

        return RequestPlan(
            remaining=statuses, mamba_on_matched_parent=fork_point
        )

    # ------------------------------------------------------------------
    # FLOP efficiency per node (original Marconi formula)
    # ------------------------------------------------------------------

    def _node_flop_efficiency_fast(
        self, seqlen_total: int, seqlen_parent: int
    ) -> float:
        """FLOP efficiency with precomputed depth values (no tree walk)."""
        seqlen_child = seqlen_total - seqlen_parent

        d, n = self.d, self.n

        flop_mamba = self.num_ssm_layers * _mamba1_flop(seqlen_child, d, n)
        flop_attn = self.num_attn_layers * (_attn_flop(seqlen_total, d) - _attn_flop(seqlen_parent, d))
        # Original bug: mlp parent term uses _attn_flop instead of _mlp_flop
        flop_mlp = self.num_mlp_layers * (_mlp_flop(seqlen_total, d) - _attn_flop(seqlen_parent, d))
        total_flop_savings = flop_mamba + flop_attn + flop_mlp

        total_memory = (
            self.num_ssm_layers * _mamba_state_size(d, n)
            + self.num_attn_layers * _kvs_size(seqlen_total, d)
        )
        if total_memory == 0:
            return 0.0
        return total_flop_savings / total_memory

    # ------------------------------------------------------------------
    # Unified eviction
    # ------------------------------------------------------------------

    def _collect_and_score(
        self, tree: RadixTree
    ) -> List[Tuple[float, RadixNode, EvictOp]]:
        """Score candidates using the original Marconi formula.

        Uses top-down BFS to compute depth_tokens and checkpoint_depth
        inline, avoiding O(depth) parent walks per node.

        score = alpha * norm_flop_eff + norm_recency
        (lower score → evict first)
        """
        current_ts = tree.clock

        # Top-down BFS: depth_tokens is precomputed on each node; we only
        # need to track checkpoint_depth (nearest ancestor with mamba state).
        candidates: List[Tuple[RadixNode, EvictOp, int, int]] = []
        q: deque[Tuple[RadixNode, int]] = deque()
        for child in tree.root.children.values():
            q.append((child, 0))

        while q:
            node, checkpoint_depth = q.popleft()

            child_checkpoint = node.depth_tokens if node.has_mamba_state else checkpoint_depth

            num_ch = len(node.children)
            if num_ch == 0:
                candidates.append((node, "leaf", node.depth_tokens, checkpoint_depth))
            elif node.has_mamba_state:
                candidates.append((node, "mamba", node.depth_tokens, checkpoint_depth))

            for ch in node.children.values():
                q.append((ch, child_checkpoint))

        if not candidates:
            return []

        # --- Recency: 1 / (current_ts - ts), higher = hotter ---
        recency_values = [
            1.0 / max(current_ts - n.last_access, 1)
            for n, _, _, _ in candidates
        ]

        # --- FLOP efficiency with precomputed depths (no parent walk) ---
        flop_eff_values = [
            self._node_flop_efficiency_fast(dt, cp)
            for _, _, dt, cp in candidates
        ]

        # --- Normalise both to [0, 1] ---
        norm_recency = _normalize(recency_values)
        norm_flop_eff = _normalize(flop_eff_values)

        # --- Score: alpha * eff + recency (lower → evict first) ---
        scored: List[Tuple[float, RadixNode, EvictOp]] = []
        for (n, op, _, _), rec, eff in zip(candidates, norm_recency, norm_flop_eff):
            score = self.alpha * eff + rec
            scored.append((score, n, op))

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
