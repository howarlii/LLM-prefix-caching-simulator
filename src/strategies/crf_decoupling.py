"""CRF-decoupling eviction strategy for hybrid prefix tree caches.

Maintains a Combined Recency and Frequency (CRF) score per node to guide
two distinct eviction operations:

* **Operation A** (partial evict): drop only the Mamba SSM state from a node,
  keeping its KV cache.  Scored by recomputation cost relative to the freed
  Mamba state memory.
* **Operation B** (chain evict): prune an entire leaf chain up to the nearest
  branching point or Mamba checkpoint.  Scored by recomputation cost of the
  whole chain relative to freed memory.

CRF update on cache hit
~~~~~~~~~~~~~~~~~~~~~~~
On every hit that touches node *n*::

    crf_new = 1.0 + 2^(-lambda * delta_t) * crf_old

where ``delta_t`` is the time since the last access.  Smaller ``lambda``
gives more weight to frequency; larger ``lambda`` favours recency.

The strategy integrates with the existing two-phase eviction loop
(``select_mamba_state_evictions`` then ``select_nodes``) but compares
per-byte value across both operations to decide whether a Mamba-state
demotion is cheaper than a chain eviction.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictionStrategy


def _depth_tokens(node: RadixNode) -> int:
    """Total tokens from root down to *node* (inclusive)."""
    d = 0
    n: Optional[RadixNode] = node
    while n is not None and n.parent is not None:
        d += len(n.page)
        n = n.parent
    return d


def _delta_mamba(node: RadixNode) -> int:
    """Tokens from the nearest ancestor with Mamba state to *node* (inclusive).

    If no ancestor carries Mamba state the gap spans from root to *node*.
    """
    total = len(node.page)
    cur = node.parent
    while cur is not None and cur.parent is not None:
        if cur.has_mamba_state:
            return total
        total += len(cur.page)
        cur = cur.parent
    return total


def _get_crf(node: RadixNode) -> float:
    return getattr(node, "_crf_value", 1.0)


def _get_crf_last_access(node: RadixNode) -> float:
    return getattr(node, "_crf_last_access", 0.0)


def _set_crf(node: RadixNode, crf: float, ts: float) -> None:
    node._crf_value = crf  # type: ignore[attr-defined]
    node._crf_last_access = ts  # type: ignore[attr-defined]


def _update_crf(node: RadixNode, current_time: float, lambda_decay: float) -> None:
    """Apply the CRF update formula."""
    old_crf = _get_crf(node)
    old_ts = _get_crf_last_access(node)
    if old_ts > 0:
        delta = current_time - old_ts
        new_crf = 1.0 + math.pow(2, -lambda_decay * delta) * old_crf
    else:
        new_crf = 1.0
    _set_crf(node, new_crf, current_time)


def _init_crf(node: RadixNode, current_time: float) -> None:
    _set_crf(node, 1.0, current_time)


class CRFDecouplingStrategy(EvictionStrategy):
    """CRF-decoupling eviction with dual operation scoring.

    Parameters
    ----------
    lambda_decay:
        CRF exponential decay rate.  Smaller = more weight on frequency.
    c_attn:
        Relative FLOPs cost per token for attention layers.
    c_ssm:
        Relative FLOPs cost per token for SSM layers.
    """

    def __init__(
        self,
        lambda_decay: float = 0.001,
        c_attn: float = 1.0,
        c_ssm: float = 1.0,
    ) -> None:
        self.lambda_decay = lambda_decay
        self.c_attn = c_attn
        self.c_ssm = c_ssm

    # ------------------------------------------------------------------
    # CRF bookkeeping
    # ------------------------------------------------------------------

    def on_cache_hit(
        self, tree: RadixTree, matched_nodes: List[RadixNode]
    ) -> None:
        ts = tree.clock
        for node in matched_nodes:
            _update_crf(node, ts, self.lambda_decay)

    def on_new_nodes_inserted(
        self, tree: RadixTree, new_nodes: List[RadixNode]
    ) -> None:
        ts = tree.clock
        for node in new_nodes:
            _init_crf(node, ts)

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def admit_mamba_state(self, node: RadixNode) -> bool:
        return True

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_a(self, node: RadixNode, mamba_token_equiv: int) -> float:
        """H_A: value score for dropping only a node's Mamba state."""
        crf = _get_crf(node)
        delta = _delta_mamba(node)
        s_m = max(mamba_token_equiv, 1)
        return crf * delta * self.c_ssm / s_m

    def _compute_chain(self, leaf: RadixNode) -> List[RadixNode]:
        """Walk up from *leaf* collecting the evictable chain.

        Stops at branching points, Mamba checkpoints, or root.
        """
        chain = [leaf]
        cur = leaf.parent
        while cur is not None and cur.parent is not None:
            if len(cur.children) > 1:
                break
            if cur.has_mamba_state:
                break
            chain.append(cur)
            cur = cur.parent
        return chain

    def _score_b(
        self, chain: List[RadixNode], mamba_token_equiv: int
    ) -> Tuple[float, int]:
        """H_B and S_chain for a chain eviction.

        Returns (score, s_chain_tokens) so the caller can compute per-byte
        value for cross-operation comparison.
        """
        max_crf = max(_get_crf(n) for n in chain)
        chain_tokens = sum(len(n.page) for n in chain)
        s_chain = chain_tokens
        for n in chain:
            if n.has_mamba_state:
                s_chain += mamba_token_equiv
        if s_chain == 0:
            return (float("inf"), 0)
        score = max_crf * chain_tokens * (self.c_attn + self.c_ssm) / s_chain
        return (score, s_chain)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def select_mamba_state_evictions(
        self, tree: RadixTree, num_states: int
    ) -> List[RadixNode]:
        """Operation A: drop Mamba state from the cheapest candidate.

        Before returning, compares against the best Operation B candidate.
        Returns empty if chain eviction would free memory more cheaply,
        letting the simulator fall through to ``select_nodes``.
        """
        mte = tree.mamba_state_token_equiv
        if mte == 0:
            return []

        candidates = tree.nodes_with_mamba_state()
        if not candidates:
            return []

        # Best Operation A
        best_a = min(candidates, key=lambda n: self._score_a(n, mte))
        score_a = self._score_a(best_a, mte)

        # Compare against best Operation B
        leaves = tree.leaf_nodes()
        if leaves:
            best_b_score = float("inf")
            best_b_s_chain = 0
            for leaf in leaves:
                chain = self._compute_chain(leaf)
                sb, sc = self._score_b(chain, mte)
                if sb < best_b_score:
                    best_b_score = sb
                    best_b_s_chain = sc

            s_m = max(mte, 1)
            value_a = score_a / s_m
            value_b = best_b_score / max(best_b_s_chain, 1) if best_b_s_chain > 0 else float("inf")

            if value_b < value_a:
                # Chain eviction is cheaper per unit — skip mamba eviction
                return []

        return [best_a]

    def select_nodes(self, tree: RadixTree, num_nodes: int) -> List[RadixNode]:
        """Operation B: evict the leaf whose chain has the lowest score."""
        leaves = tree.leaf_nodes()
        if not leaves:
            return []

        mte = tree.mamba_state_token_equiv
        scored: List[Tuple[float, RadixNode]] = []
        for leaf in leaves:
            chain = self._compute_chain(leaf)
            sb, _ = self._score_b(chain, mte)
            scored.append((sb, leaf))

        scored.sort(key=lambda x: x[0])
        return [n for _, n in scored[:num_nodes]]
