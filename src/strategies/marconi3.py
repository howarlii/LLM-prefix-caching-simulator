"""Marconi3: Marconi-base eviction with mid-chain checkpointing and multiple scoring modes.

Built on top of Marconi (v1) with two additions:

1. **Mid-chain Mamba state placement** (from Marconi2): When a new leaf chain
   is inserted and its total span (from last checkpoint/root) >= 2048 tokens,
   a Mamba state is placed at roughly 55% of that span.

2. **Multiple eviction score formulas** controlled by ``evict_mode``:

   - ``ev0`` (default): Original Marconi scoring.
     ``score = alpha * norm_flop_eff + norm_recency``
     where ``flop_eff = total_flop_savings / total_memory`` and
     ``recency = 1 / (current_ts - last_access)``, both min-max normalised.

   - ``ev1``: Uses the same FLOP formulas for depth computation, but scores
     with marconi2-style normalisation on raw recency and depth:
     ``norm_r = (r - min_r) / (max_r - min_r)``
     ``norm_d = (d - min_d) / (max_d - min_d)``
     ``score = norm_r + alpha * norm_d``
     where ``r = last_access`` and ``d = flop_efficiency``.

   - ``ev2``: Uses raw FLOP savings (not divided by memory), normalised,
     then divided by freed capacity:
     ``score = (norm_r + alpha * norm_flop) / freed``
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
    compute_min_mamba_admit_depth,
)
from src.model_config import (
    DEFAULT_MODEL,
    ModelConfig,
    _attn_flop,
    _kvs_size,
    _mamba1_flop,
    _mamba_state_size,
    _mlp_flop,
)
from src.strategies.marconi import _normalize

# Minimum chain token length to place a mid-chain checkpoint.
_MIN_CHAIN_TOKENS_FOR_MID_CHECKPOINT = 2048


def _effective_ts(node: RadixNode) -> int:
    """Recency sort key: prefer the strategy-managed timestamp, fall back to last_access.

    When ``newtouch`` is disabled, ``_m3_lru`` is never written and this
    collapses to ``node.last_access`` — identical to the pre-newtouch
    scoring behaviour.
    """
    return getattr(node, "_m3_lru", node.last_access)


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
        If True, place a mid-chain mamba state at ~55% of long new chains.
        Defaults to False (mid-chain checkpointing disabled).
    newtouch:
        If False (default), recency is taken directly from
        ``node.last_access`` — the radix tree refreshes every matched
        node on a hit, matching standard LRU-style behaviour.

        If True, the strategy overrides the touch policy via a parallel
        ``_m3_lru`` attribute maintained on each node:

        * The deepest matched node (last hit) is always refreshed.
        * For each ancestor matched node ``n``, look at the "previous
          checkpoint" — the deepest mamba-state node already passed
          while walking upward from the deepest hit toward ``n``:

          - leaf checkpoint (no children)  → do not refresh ``n``;
          - branch-point checkpoint (≥2 children) → refresh ``n``;
          - otherwise (no checkpoint, or single-child checkpoint)
            → do not refresh ``n``.

        The eviction scoring formulas are unchanged — they simply read
        ``_effective_ts(n)`` instead of ``n.last_access`` for recency,
        and fall back to ``last_access`` when ``_m3_lru`` is not set
        (e.g. split-induced suffixes).
    model:
        Model architecture configuration.  Provides layer counts, hidden
        dimension, and SSM state dimension for FLOP computation.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        evict_mode: str = "ev0",
        use_mid_chain_checkpoint: bool = False,
        newtouch: bool = False,
        model: ModelConfig = DEFAULT_MODEL,
        gpu_flops: Optional[float] = None,
        pcie_bandwidth: Optional[float] = None,
    ) -> None:
        self.alpha = alpha
        self.evict_mode = evict_mode
        self.use_mid_chain_checkpoint = use_mid_chain_checkpoint
        self.newtouch = newtouch
        self.model = model
        self.num_ssm_layers = model.num_ssm_layers
        self.num_attn_layers = model.num_attn_layers
        self.num_mlp_layers = model.num_mlp_layers
        self.d = model.d_model
        self.n = model.ssm_state_dim
        self.gpu_flops = gpu_flops
        self.pcie_bandwidth = pcie_bandwidth
        # Hardware-aware mamba-state admit depth: nodes shallower than this
        # are not worth a Mamba state because loading the state via PCIe
        # would cost more than just recomputing the prefix from scratch.
        self._min_mamba_admit_depth = compute_min_mamba_admit_depth(
            model, gpu_flops, pcie_bandwidth
        )

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def plan_request(
        self,
        tree: RadixTree,
        matched_nodes: List[RadixNode],
        remaining_pages: List[PageKey],
    ) -> RequestPlan:
        """Admit every remaining page; place mamba at turn-end (if deep
        enough), fork-point parent, and optionally a mid-chain checkpoint."""
        if not remaining_pages:
            return RequestPlan(remaining=[])

        statuses: List[PageStatus] = [PageStatus.KV_ONLY] * len(remaining_pages)
        fork_point = False

        if tree.mamba_state_token_equiv == 0:
            return RequestPlan(remaining=statuses)

        parent = matched_nodes[-1] if matched_nodes else tree.root
        parent_depth = parent.depth_tokens
        chain_tokens = sum(len(p) for p in remaining_pages)
        end_depth = parent_depth + chain_tokens
        threshold = self._min_mamba_admit_depth

        # --- Turn-end mamba ---
        if threshold == 0 or end_depth >= threshold:
            statuses[-1] = PageStatus.KV_AND_MAMBA

        # --- Fork-point parent ---
        if (
            parent is not tree.root
            and len(parent.children) >= 1
            and not parent.has_mamba_state
            and (threshold == 0 or parent_depth >= threshold)
        ):
            fork_point = True

        # --- Mid-chain checkpoint at ~55% of (last_checkpoint → end) ---
        if not self.use_mid_chain_checkpoint:
            return RequestPlan(
                remaining=statuses, mamba_on_matched_parent=fork_point
            )

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
        if (
            target_depth > parent_depth
            and target_depth < end_depth
            and (threshold == 0 or target_depth >= threshold)
        ):
            offset = target_depth - parent_depth
            cum = 0
            for i, p in enumerate(remaining_pages):
                cum += len(p)
                if cum >= offset:
                    if statuses[i] != PageStatus.KV_AND_MAMBA:
                        statuses[i] = PageStatus.KV_AND_MAMBA
                    break

        return RequestPlan(
            remaining=statuses, mamba_on_matched_parent=fork_point
        )

    # ------------------------------------------------------------------
    # Bookkeeping (newtouch LRU + selective ancestor refresh)
    # ------------------------------------------------------------------

    def on_cache_hit(
        self, tree: RadixTree, matched_nodes: List[RadixNode]
    ) -> None:
        """Selective touch policy when ``newtouch`` is enabled.

        Does nothing when ``newtouch`` is False — the radix tree's
        built-in ``last_access`` refresh is sufficient, and
        ``_effective_ts`` falls back to it.
        """
        if not self.newtouch or not matched_nodes:
            return
        ts = tree.clock

        for i in range(len(matched_nodes) - 1, -1, -1):
            n = matched_nodes[i]
            if n.has_mamba_state:
                n._m3_lru = ts  # type: ignore[attr-defined]
                break

    def on_nodes_inserted(
        self, tree: RadixTree, new_nodes: List[RadixNode]
    ) -> None:
        """Initialise the per-node ``_m3_lru`` timestamp on new nodes."""
        if not self.newtouch or not new_nodes:
            return
        ts = tree.clock
        for n in new_nodes:
            n._m3_lru = ts  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # FLOP / memory helpers
    # ------------------------------------------------------------------

    def _prefill_flop(self, seqlen: int) -> float:
        """Total prefill FLOP count from position 0 to *seqlen*.

        Accounts for the O(L^2) nature of attention.
        """
        d, n = self.d, self.n
        return (
            self.num_ssm_layers * _mamba1_flop(seqlen, d, n)
            + self.num_attn_layers * _attn_flop(seqlen, d)
            + self.num_mlp_layers * _mlp_flop(seqlen, d)
        )

    def _node_flop_savings(self, depth_tokens: int, checkpoint_depth: int) -> float:
        """FLOP count saved by caching from *checkpoint_depth* to *depth_tokens*.

        Equal to prefill_flop(depth_tokens) - prefill_flop(checkpoint_depth).
        """
        return self._prefill_flop(depth_tokens) - self._prefill_flop(checkpoint_depth)

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

        # Top-down BFS: depth_tokens is precomputed on each node; only
        # checkpoint_depth (nearest mamba ancestor) needs tracking.
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

        flop_eff = flop_savings / memory_occupy per node.
        """
        recency_values = [
            1.0 / max(current_ts - _effective_ts(n), 1)
            for n, _, _, _ in candidates
        ]

        flop_eff_values = []
        for n, op, dt, cp in candidates:
            flop = self._node_flop_savings(dt, cp)
            if op == "leaf":
                mem = self._node_memory_occupy(dt, cp, n.has_mamba_state)
            elif op == "mamba" and len(n.children) == 1:
                mem = self._mamba_memory_occupy()
            else:
                mem = 0
            flop_eff_values.append(flop / mem if mem > 0 else 0.0)

        norm_recency = _normalize(recency_values)
        norm_flop_eff = _normalize(flop_eff_values)

        scored: List[Tuple[float, RadixNode, EvictOp]] = []
        for (n, op, _, _), rec, eff in zip(candidates, norm_recency, norm_flop_eff):
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

        flop_eff = flop_savings / memory_occupy.
        score = norm_r + alpha * norm_d
        """
        recency_values = [
            1.0 / max(current_ts - _effective_ts(n), 1)
            for n, _, _, _ in candidates
        ]

        flop_eff_values = []
        for n, op, dt, cp in candidates:
            flop = self._node_flop_savings(dt, cp)
            if op == "leaf":
                mem = self._node_memory_occupy(dt, cp, n.has_mamba_state)
            else:
                mem = self._mamba_memory_occupy()
            flop_eff_values.append(flop / mem if mem > 0 else 0.0)

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

        raw_score = norm_r + alpha * norm_flop
        score = raw_score / memory_occupy
        """
        recencies = [_effective_ts(n) for n, _, _, _ in candidates]

        flop_values = [
            self._node_flop_savings(dt, cp)
            for _, _, dt, cp in candidates
        ]

        mem_values = [
            self._node_memory_occupy(dt, cp, n.has_mamba_state)
            for n, _, dt, cp in candidates
        ]

        min_r, max_r = min(recencies), max(recencies)
        min_f, max_f = min(flop_values), max(flop_values)

        scored: List[Tuple[float, RadixNode, EvictOp]] = []
        for (n, op, _, _), r, f, mem in zip(candidates, recencies, flop_values, mem_values):
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

        raw_score = norm_r + alpha * norm_flop
        score = raw_score / memory_occupy
        """
        recencies = [_effective_ts(n) for n, _, _, _ in candidates]

        flop_values = [
            self._node_flop_savings(dt, cp)
            for _, _, dt, cp in candidates
        ]

        mem_values = [
            self._node_memory_occupy(dt, cp, n.has_mamba_state)
            for n, _, dt, cp in candidates
        ]


        norm_recency = _normalize(recencies, 0)
        norm_flop = _normalize(flop_values, 0)
        min_r, max_r = min(recencies), max(recencies)
        min_f, max_f = min(flop_values), max(flop_values)

        scored: List[Tuple[float, RadixNode, EvictOp]] = []
        for (n, op, _, _), norm1_r, r, norm1_f, f, mem in zip(candidates, norm_recency, recencies, norm_flop, flop_values, mem_values):
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
