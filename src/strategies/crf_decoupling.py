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
On every hit that touches node *n* **with a Mamba state**::

    crf_new = 1.0 + 2^(-lambda * delta_t) * crf_old

where ``delta_t`` is the time since the last access.  Smaller ``lambda``
gives more weight to frequency; larger ``lambda`` favours recency.

Nodes without a Mamba state do not maintain CRF scores — they cannot
save computation on their own, so tracking their popularity is pointless.

The strategy uses unified ``select_eviction`` to compare per-byte value
across both operations.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

from src.radix_tree import RadixNode, RadixTree
from src.strategies.base import EvictOp, EvictionStrategy

_DEFAULT_CRF = 2.0


def _delta_mamba(node: RadixNode) -> int:
    """Tokens from the nearest ancestor with Mamba state to *node* (inclusive).

    If no ancestor carries Mamba state the gap spans from root to *node*.
    """
    total = node.num_tokens
    cur = node.parent
    while cur is not None and cur.parent is not None:
        if cur.has_mamba_state:
            return total
        total += cur.num_tokens
        cur = cur.parent
    return total


def _get_crf(node: RadixNode) -> float:
    return getattr(node, "_crf_value", _DEFAULT_CRF)


def _get_crf_ts(node: RadixNode) -> int:
    return getattr(node, "_crf_ts", 0)


def _effective_crf(node: RadixNode, t_now: int, lambda_decay: float) -> float:
    """Return CRF with time-decay applied up to *t_now*."""
    stored = _get_crf(node)
    ts = _get_crf_ts(node)
    if ts > 0 and t_now > ts:
        v = math.pow(2, -lambda_decay * (t_now - ts)) * stored
        _set_crf(node, v, ts)  # update in-place with decay
    return stored


def _set_crf(node: RadixNode, crf: float, ts: int) -> None:
    node._crf_value = crf  # type: ignore[attr-defined]
    node._crf_ts = ts  # type: ignore[attr-defined]


def _update_crf(node: RadixNode, ts: int, lambda_decay: float) -> None:
    old_crf = _get_crf(node)
    old_ts = _get_crf_ts(node)
    if old_ts > 0:
        delta = ts - old_ts
        new_crf = 1.0 + math.pow(2, -lambda_decay * delta) * old_crf
    else:
        new_crf = 1.0
    _set_crf(node, new_crf, ts)


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
        lambda_decay: float = 0.5,
        c_attn: float = 1.0,
        c_ssm: float = 1.0,
    ) -> None:
        self.lambda_decay = lambda_decay
        self.c_attn = c_attn
        self.c_ssm = c_ssm

    # ------------------------------------------------------------------
    # CRF bookkeeping — only for nodes carrying Mamba state
    # ------------------------------------------------------------------

    def on_cache_hit(
        self, tree: RadixTree, matched_nodes: List[RadixNode]
    ) -> None:
        if tree.mamba_state_token_equiv == 0:
            return
        ts = tree.clock
        ld = self.lambda_decay
        for node in matched_nodes:
            if node.has_mamba_state:
                _update_crf(node, ts, ld)
                break

    def on_nodes_inserted(
        self, tree: RadixTree, new_nodes: List[RadixNode]
    ) -> None:
        """Initialise CRF on new nodes that received mamba state."""
        if tree.mamba_state_token_equiv == 0:
            return
        ts = tree.clock
        for node in new_nodes:
            if node.has_mamba_state:
                _set_crf(node, _DEFAULT_CRF, ts)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_a(self, node: RadixNode, mamba_token_equiv: int, t_now: int) -> float:
        """H_A: value score for dropping only a node's Mamba state."""
        crf = _effective_crf(node, t_now, self.lambda_decay)
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

    def _score_b_chain(
        self, chain: List[RadixNode], mamba_token_equiv: int, t_now: int
    ) -> Tuple[float, int]:
        """H_B and S_chain for a chain eviction.

        CRF is only read from nodes with Mamba state; KV-only nodes in the
        chain use the default score.
        """
        max_crf = 0.0
        has_mamba = False
        chain_tokens = 0
        s_chain = 0
        for n in chain:
            toks = n.num_tokens
            chain_tokens += toks
            s_chain += toks
            if n.has_mamba_state:
                has_mamba = True
                crf = _effective_crf(n, t_now, self.lambda_decay)
                if crf > max_crf:
                    max_crf = crf
                s_chain += mamba_token_equiv
        if not has_mamba:
            max_crf = _DEFAULT_CRF
        if s_chain == 0:
            return (float("inf"), 0)
        score = max_crf * chain_tokens * (self.c_attn + self.c_ssm) / s_chain
        return (score, s_chain)

    # ------------------------------------------------------------------
    # Unified eviction
    # ------------------------------------------------------------------

    def select_eviction(
        self, tree: RadixTree
    ) -> Optional[Tuple[RadixNode, EvictOp]]:
        """Unified eviction: compare Operation A (mamba demotion) vs
        Operation B (chain eviction) and pick the cheapest."""
        mte = tree.mamba_state_token_equiv
        t_now = tree.clock

        leaves: List[RadixNode] = list(tree.leaf_node_set())
        mamba_candidates: List[RadixNode] = list(tree.mamba_state_node_set())

        # Best Operation A: mamba-state demotion (non-branching nodes only)
        best_a: Optional[RadixNode] = None
        best_a_value = float("inf")
        if mte > 0:
            valid = [n for n in mamba_candidates if len(n.children) <= 1]
            if not valid and mamba_candidates:
                warnings.warn(
                    "CRFDecoupling: all mamba-state candidates are branching "
                    "points (>1 child); skipping mamba-state-only eviction.",
                    stacklevel=2,
                )
            s_m = max(mte, 1)
            for n in valid:
                score = self._score_a(n, mte, t_now)
                value = score / s_m
                if value < best_a_value:
                    best_a_value = value
                    best_a = n

        # Best Operation B: chain eviction
        best_b_leaf: Optional[RadixNode] = None
        best_b_value = float("inf")
        for leaf in leaves:
            chain = self._compute_chain(leaf)
            sb, sc = self._score_b_chain(chain, mte, t_now)
            value = sb / max(sc, 1)
            if value < best_b_value:
                best_b_value = value
                best_b_leaf = leaf

        # Pick the cheapest
        if best_a is not None and best_a_value <= best_b_value:
            return (best_a, "mamba")
        if best_b_leaf is not None:
            return (best_b_leaf, "leaf")
        return None
