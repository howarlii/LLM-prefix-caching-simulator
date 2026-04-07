"""Marconi3: Marconi-base eviction with mid-chain checkpointing and multiple scoring modes.

Built on top of Marconi (v1) with two additions:

1. **Mid-chain Mamba state placement** (from Marconi2): When a new leaf chain
   is inserted and its total span (from last checkpoint/root) >= 2048 tokens,
   a Mamba state is placed at roughly 55% of that span.

2. **Multiple eviction score formulas** controlled by ``evict_mode``:

   - ``ev0`` (default): Original Marconi scoring.
     ``score = alpha * norm_flop_eff + norm_recency``
     where ``flop_eff = total_flops_savings / total_memory`` and
     ``recency = 1 / (current_ts - last_access)``, both min-max normalised.

   - ``ev1``: Uses the same FLOP formulas for depth computation, but scores
     with marconi2-style normalisation on raw recency and depth:
     ``norm_r = (r - min_r) / (max_r - min_r)``
     ``norm_d = (d - min_d) / (max_d - min_d)``
     ``score = norm_r + alpha * norm_d``
     where ``r = last_access`` and ``d = flop_efficiency``.

   - ``ev2``: Uses raw FLOP savings (not divided by memory), normalised,
     then divided by freed capacity:
     ``score = (norm_r + alpha * norm_flops) / freed``
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictOp, EvictionStrategy
from src.model_config import (
    DEFAULT_MODEL,
    ModelConfig,
    _attn_flops,
    _kvs_size,
    _mamba1_flops,
    _mamba_state_size,
    _mlp_flops,
)
from src.strategies.marconi import _normalize

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


class Marconi3Strategy(EvictionStrategy):
    """Marconi-base eviction with mid-chain checkpointing and multiple scoring modes.

    Parameters
    ----------
    alpha:
        Weight for the FLOP-efficiency term in the eviction score.
    evict_mode:
        Scoring formula: ``"ev0"`` for original Marconi, ``"ev1"`` for
        marconi2-style normalisation with FLOP formulas.
    use_mid_chain_checkpoint:
        If True (default), place a mid-chain mamba state at ~55% of long
        new chains.  Setting False disables this behaviour.
    model:
        Model architecture configuration.  Provides layer counts, hidden
        dimension, and SSM state dimension for FLOP computation.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        evict_mode: str = "ev0",
        use_mid_chain_checkpoint: bool = True,
        model: ModelConfig = DEFAULT_MODEL,
    ) -> None:
        self.alpha = alpha
        self.evict_mode = evict_mode
        self.use_mid_chain_checkpoint = use_mid_chain_checkpoint
        self.model = model
        self.num_ssm_layers = model.num_ssm_layers
        self.num_attn_layers = model.num_attn_layers
        self.num_mlp_layers = model.num_mlp_layers
        self.d = model.d_model
        self.n = model.ssm_state_dim

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
        """Handle fork-point parent admission + mid-chain checkpoint placement."""
        if not new_nodes:
            return

        # --- Fork-point parent ---
        parent = new_nodes[0].parent
        if parent is not None and parent is not tree.root:
            if len(parent.children) > 1 and not parent.has_mamba_state:
                tree.set_mamba_state(parent)

        # --- Mid-chain checkpoint at ~55% ---
        if not self.use_mid_chain_checkpoint:
            return

        anchor = new_nodes[0].parent
        tokens_before_chain = 0
        cur = anchor
        while cur is not None and cur.parent is not None:
            if cur.has_mamba_state:
                break
            tokens_before_chain += cur.num_tokens
            cur = cur.parent

        chain_tokens = sum(n.num_tokens for n in new_nodes)
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
                remaining -= candidate.num_tokens
                if remaining <= 0:
                    break
                candidate = candidate.parent
            if candidate is not None and candidate is not tree.root and not candidate.has_mamba_state:
                if candidate.num_pages > 1:
                    offset_in_node = candidate.num_tokens + remaining
                    pages_to_keep = 0
                    cum = 0
                    for p in candidate.pages:
                        cum += len(p)
                        pages_to_keep += 1
                        if cum >= offset_in_node:
                            break
                    if 0 < pages_to_keep < candidate.num_pages:
                        tree.split_node(candidate, pages_to_keep)
                tree.set_mamba_state(candidate)
        else:
            offset_in_chain = target_pos - tokens_before_chain
            cumulative = 0
            for node in new_nodes:
                if cumulative + node.num_tokens >= offset_in_chain:
                    offset_in_node = offset_in_chain - cumulative
                    if offset_in_node > 0 and node.num_pages > 1:
                        pages_to_keep = 0
                        cum = 0
                        for p in node.pages:
                            cum += len(p)
                            pages_to_keep += 1
                            if cum >= offset_in_node:
                                break
                        if 0 < pages_to_keep < node.num_pages:
                            tree.split_node(node, pages_to_keep)
                    if not node.has_mamba_state:
                        tree.set_mamba_state(node)
                    break
                cumulative += node.num_tokens

    # ------------------------------------------------------------------
    # FLOP / memory helpers
    # ------------------------------------------------------------------

    def _prefill_flops(self, seqlen: int) -> float:
        """Total prefill FLOPs from position 0 to *seqlen*.

        Accounts for the O(L^2) nature of attention.
        """
        d, n = self.d, self.n
        return (
            self.num_ssm_layers * _mamba1_flops(seqlen, d, n)
            + self.num_attn_layers * _attn_flops(seqlen, d)
            + self.num_mlp_layers * _mlp_flops(seqlen, d)
        )

    def _node_flops_savings(self, depth_tokens: int, checkpoint_depth: int) -> float:
        """FLOPs saved by caching from *checkpoint_depth* to *depth_tokens*.

        Equal to prefill_flops(depth_tokens) - prefill_flops(checkpoint_depth).
        """
        return self._prefill_flops(depth_tokens) - self._prefill_flops(checkpoint_depth)

    def _mamba_memory_occupy(self) -> float:
        d, n = self.d, self.n
        return self.num_ssm_layers * _mamba_state_size(d, n)

    def _node_memory_occupy(
        self, depth_tokens: int, checkpoint_depth: int, has_mamba: bool
    ) -> float:
        """Memory occupied: KV cache from checkpoint to current node + mamba state if present."""
        d, n = self.d, self.n
        kv_tokens = depth_tokens - checkpoint_depth
        mem = self.num_attn_layers * _kvs_size(kv_tokens, d)
        if has_mamba:
            mem += self._mamba_memory_occupy()
        return mem

    # ------------------------------------------------------------------
    # Eviction scoring
    # ------------------------------------------------------------------

    def _collect_and_score(
        self, tree: RadixTree
    ) -> List[Tuple[float, RadixNode, EvictOp]]:
        """Score candidates and return sorted list (ascending by score).

        Dispatches to ev0 or ev1 based on ``self.evict_mode``.
        """
        current_ts = tree.clock

        # Top-down BFS to compute depth_tokens and checkpoint_depth.
        candidates: List[Tuple[RadixNode, EvictOp, int, int]] = []
        q: deque[Tuple[RadixNode, int, int]] = deque()
        for child in tree.root.children.values():
            q.append((child, child.num_tokens, 0))

        while q:
            node, depth_tokens, checkpoint_depth = q.popleft()

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

        if self.evict_mode == "ev1":
            return self._score_ev1(candidates, current_ts)
        elif self.evict_mode == "ev2":
            return self._score_ev2(candidates, current_ts, tree)
        elif self.evict_mode == "ev3":
            return self._score_ev3(candidates, current_ts, tree)
        else:
            return self._score_ev0(candidates, current_ts)

    def _score_ev0(
        self,
        candidates: List[Tuple[RadixNode, EvictOp, int, int]],
        current_ts: int,
    ) -> List[Tuple[float, RadixNode, EvictOp]]:
        """Original Marconi scoring: alpha * norm_flop_eff + norm_recency.

        flop_eff = flops_savings / memory_occupy per node.
        """
        recency_values = [
            1.0 / max(current_ts - n.last_access, 1)
            for n, _, _, _ in candidates
        ]

        flop_eff_values = []
        for n, _, dt, cp in candidates:
            flops = self._node_flops_savings(dt, cp)
            mem = self._node_memory_occupy(dt, cp, n.has_mamba_state)
            assert flops >= 0 and mem >= 0, f"Negative flops ({flops}) or memory ({mem}) for node {n}"
            flop_eff_values.append(flops / mem if mem > 0 else 0.0)

        norm_recency = _normalize(recency_values)
        norm_flop_eff = _normalize(flop_eff_values)

        scored: List[Tuple[float, RadixNode, EvictOp]] = []
        for (n, op, _, _), rec, eff in zip(candidates, norm_recency, norm_flop_eff):
            if op == "leaf":
                score = self.alpha * eff + rec
                scored.append((score, n, op))

        scored.sort(key=lambda x: x[0])
        return scored

    def _score_ev1(
        self,
        candidates: List[Tuple[RadixNode, EvictOp, int, int]],
        current_ts: int,
    ) -> List[Tuple[float, RadixNode, EvictOp]]:
        """Marconi2-style normalisation with FLOP efficiency as depth metric.

        flop_eff = flops_savings / memory_occupy.
        score = norm_r + alpha * norm_d
        """
        recency_values = [
            1.0 / max(current_ts - n.last_access, 1)
            for n, _, _, _ in candidates
        ]

        flop_eff_values = []
        for n, op, dt, cp in candidates:
            flops = self._node_flops_savings(dt, cp)
            if op == "leaf":
                mem = self._node_memory_occupy(dt, cp, n.has_mamba_state)
            else:
                mem = self._mamba_memory_occupy()
            flop_eff_values.append(flops / mem if mem > 0 else 0.0)

        norm_recency = _normalize(recency_values)
        norm_flop_eff = _normalize(flop_eff_values)

        scored: List[Tuple[float, RadixNode, EvictOp]] = []
        for (n, op, _, _), rec, eff in zip(candidates, norm_recency, norm_flop_eff):
            score = self.alpha * eff + rec
            scored.append((score, n, op))

        scored.sort(key=lambda x: x[0])
        return scored

    def _score_ev2(
        self,
        candidates: List[Tuple[RadixNode, EvictOp, int, int]],
        current_ts: int,
        tree: RadixTree,
    ) -> List[Tuple[float, RadixNode, EvictOp]]:
        """Raw FLOP savings normalised, then divided by memory occupy.

        raw_score = norm_r + alpha * norm_flops
        score = raw_score / memory_occupy
        """
        recencies = [n.last_access for n, _, _, _ in candidates]

        flops_values = [
            self._node_flops_savings(dt, cp)
            for _, _, dt, cp in candidates
        ]

        mem_values = [
            self._node_memory_occupy(dt, cp, n.has_mamba_state)
            for n, _, dt, cp in candidates
        ]

        min_r, max_r = min(recencies), max(recencies)
        min_f, max_f = min(flops_values), max(flops_values)

        scored: List[Tuple[float, RadixNode, EvictOp]] = []
        for (n, op, _, _), r, f, mem in zip(candidates, recencies, flops_values, mem_values):
            norm_r = (r - min_r) / (max_r - min_r) if max_r > min_r else 0.0
            norm_f = (f - min_f) / (max_f - min_f) if max_f > min_f else 0.0
            raw_score = norm_r + self.alpha * norm_f
            assert mem > 0, f"Memory occupy is zero for node {n}, cannot divide by zero in ev2 scoring."
            scored.append((raw_score / mem, n, op))

        scored.sort(key=lambda x: x[0])
        return scored

    def _score_ev3(
        self,
        candidates: List[Tuple[RadixNode, EvictOp, int, int]],
        current_ts: int,
        tree: RadixTree,
    ) -> List[Tuple[float, RadixNode, EvictOp]]:
        """Raw FLOP savings normalised, then divided by memory occupy.

        raw_score = norm_r + alpha * norm_flops
        score = raw_score / memory_occupy
        """
        recencies = [n.last_access for n, _, _, _ in candidates]

        flops_values = [
            self._node_flops_savings(dt, cp)
            for _, _, dt, cp in candidates
        ]

        mem_values = [
            self._node_memory_occupy(dt, cp, n.has_mamba_state)
            for n, _, dt, cp in candidates
        ]


        norm_recency = _normalize(recencies, 0)
        norm_flop = _normalize(flops_values, 0)
        min_r, max_r = min(recencies), max(recencies)
        min_f, max_f = min(flops_values), max(flops_values)

        scored: List[Tuple[float, RadixNode, EvictOp]] = []
        for (n, op, _, _), norm1_r, r, norm1_f, f, mem in zip(candidates, norm_recency, recencies, norm_flop, flops_values, mem_values):
            norm_r = (r - min_r) / (max_r - min_r) if max_r > min_r else 0.0
            norm_f = (f - min_f) / (max_f - min_f) if max_f > min_f else 0.0
            if abs(norm_r - norm1_r) > 1e-6 or abs(norm_f - norm1_f) > 1e-6:
                print(f"Warning: Normalisation mismatch for node {n}. norm_r: {norm_r} vs {norm1_r}, norm_f: {norm_f} vs {norm1_f}")
                assert False, f"Normalisation mismatch between _normalize and manual min-max for node {n}"
            raw_score = norm_r + self.alpha * norm_f
            assert mem > 0, f"Memory occupy is zero for node {n}, cannot divide by zero in ev2 scoring."
            scored.append((raw_score / mem, n, op))

        scored.sort(key=lambda x: x[0])
        return scored

    # ------------------------------------------------------------------
    # Eviction entry point
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
