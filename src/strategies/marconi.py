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
  the MLP bug (uses ``get_attn_flops`` instead of ``get_mlp_flops`` for
  the parent subtraction term).

Candidate operations:
* Leaf nodes (``len(children) == 0``): removed entirely.
* Single-child nodes with Mamba state: Mamba state demoted.
* Single-child nodes without Mamba state: treated as leaf removal.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictOp, EvictionStrategy


# ---------------------------------------------------------------------------
# FLOP / memory helpers (matching marconi/utils.py exactly)
# ---------------------------------------------------------------------------

def _attn_flops(l: int, d: int) -> float:
    """Attention block FLOPs: 8·L·D² + 4·L²·D."""
    return 8 * l * d ** 2 + 4 * l ** 2 * d


def _mlp_flops(l: int, d: int) -> float:
    """MLP block FLOPs: 16·L·D²."""
    return 16 * l * d ** 2


def _mamba1_flops(l: int, d: int, n: int) -> float:
    """Mamba-1 layer FLOPs: 12·L·D² + 16·L·D·N + 10·L·D."""
    return 12 * l * d ** 2 + 16 * l * d * n + 10 * l * d


def _kvs_size(l: int, d: int) -> float:
    """KV cache size in bytes for one attention layer: 2·L·D·2."""
    return 2 * l * d * 2


def _mamba_state_size(d: int, n: int, conv_kernel: int = 4, expand: int = 2) -> float:
    """SSM + conv state size in bytes for one SSM layer."""
    return d * n * 2 + (expand * d + 2 * n) * conv_kernel * 2



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
    num_ssm_layers, num_attn_layers, num_mlp_layers:
        Model layer counts for FLOP computation.
    d:
        Model hidden dimension.
    n:
        SSM state dimension.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        num_ssm_layers: int = 48,
        num_attn_layers: int = 16,
        num_mlp_layers: int = 64,
        d: int = 4096,
        n: int = 128,
    ) -> None:
        self.alpha = alpha
        self.num_ssm_layers = num_ssm_layers
        self.num_attn_layers = num_attn_layers
        self.num_mlp_layers = num_mlp_layers
        self.d = d
        self.n = n

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
    # FLOP efficiency per node (original Marconi formula)
    # ------------------------------------------------------------------

    def _node_flops_efficiency_fast(
        self, seqlen_total: int, seqlen_parent: int
    ) -> float:
        """FLOP efficiency with precomputed depth values (no tree walk)."""
        seqlen_child = seqlen_total - seqlen_parent

        d, n = self.d, self.n

        flops_mamba = self.num_ssm_layers * _mamba1_flops(seqlen_child, d, n)
        flops_attn = self.num_attn_layers * (_attn_flops(seqlen_total, d) - _attn_flops(seqlen_parent, d))
        # Original bug: mlp parent term uses _attn_flops instead of _mlp_flops
        flops_mlp = self.num_mlp_layers * (_mlp_flops(seqlen_total, d) - _attn_flops(seqlen_parent, d))
        total_flops_savings = flops_mamba + flops_attn + flops_mlp

        total_memory = (
            self.num_ssm_layers * _mamba_state_size(d, n)
            + self.num_attn_layers * _kvs_size(seqlen_total, d)
        )
        if total_memory == 0:
            return 0.0
        return total_flops_savings / total_memory

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

        # Top-down BFS: compute depth_tokens and checkpoint_depth as we go.
        # Each queue entry: (node, depth_tokens, checkpoint_depth)
        # checkpoint_depth = depth_tokens of nearest ancestor with mamba state (0 = root)
        candidates: List[Tuple[RadixNode, EvictOp, int, int]] = []
        q: deque[Tuple[RadixNode, int, int]] = deque()
        for child in tree.root.children.values():
            q.append((child, child.num_tokens, 0))

        while q:
            node, depth_tokens, checkpoint_depth = q.popleft()

            # Checkpoint depth for *children* of this node
            child_checkpoint = depth_tokens if node.has_mamba_state else checkpoint_depth

            num_ch = len(node.children)
            if num_ch == 0:
                candidates.append((node, "leaf", depth_tokens, checkpoint_depth))
            elif node.has_mamba_state:
                candidates.append((node, "mamba", depth_tokens, checkpoint_depth))

            for ch in node.children.values():
                q.append((ch, depth_tokens + ch.num_tokens, child_checkpoint))

        if not candidates:
            return []

        # --- Recency: 1 / (current_ts - ts), higher = hotter ---
        recency_values = [
            1.0 / max(current_ts - n.last_access, 1)
            for n, _, _, _ in candidates
        ]

        # --- FLOP efficiency with precomputed depths (no parent walk) ---
        flop_eff_values = [
            self._node_flops_efficiency_fast(dt, cp)
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
