"""KV prefix cache simulator with radix tree and pluggable eviction / admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.radix_tree import PageKey, RadixTree
from src.strategies.base import EvictionStrategy


def tokens_to_pages(tokens: List[int], page_size: int) -> List[PageKey]:
    """Split token ids into pages; last page may be shorter than ``page_size``."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    pages: List[PageKey] = []
    for i in range(0, len(tokens), page_size):
        pages.append(tuple(tokens[i : i + page_size]))
    return pages


@dataclass
class RequestTrace:
    """Per-request statistics."""

    input_tokens: int
    hit_tokens: int
    hit_pages: int
    total_pages: int
    turn_hit_tokens: int
    # Pages whose KV cache was present but had no Mamba state coverage —
    # they are in the tree but do not yield compute savings in hybrid mode.
    # Always 0 for pure full-attention (mamba_state_token_equiv == 0).
    kv_only_hit_pages: int = 0

    @property
    def miss_tokens(self) -> int:
        return self.input_tokens - self.hit_tokens

    @property
    def per_request_token_hit_rate(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return self.hit_tokens / self.input_tokens


@dataclass
class SimulationState:
    """Accumulated counters and usage samples for one simulation run."""

    traces: List[RequestTrace] = field(default_factory=list)
    usage_samples: List[int] = field(default_factory=list)

    def record_trace(self, t: RequestTrace, cached_tokens_after: int) -> None:
        self.traces.append(t)
        self.usage_samples.append(cached_tokens_after)


class KVCacheSimulator:
    """Processes tokenized requests through a page-based radix cache.

    Parameters
    ----------
    page_size:
        Number of tokens per cache page.
    strategy:
        Eviction (and optionally admission) strategy.
    capacity_tokens:
        Maximum effective token-equivalent units the cache may hold.
        ``None`` = unlimited.  In hybrid mode this budget includes Mamba state
        overhead (each state counts as ``mamba_state_token_equiv`` tokens).
    mamba_state_token_equiv:
        How many token-equivalent units one Mamba SSM state occupies relative
        to full-attention KV cache.  ``0`` (default) = pure full-attention
        mode; no Mamba state is stored and cache hits are not gated by Mamba
        state availability.
    """

    def __init__(
        self,
        page_size: int,
        strategy: EvictionStrategy,
        capacity_tokens: Optional[int] = None,
        mamba_state_token_equiv: int = 0,
    ) -> None:
        self.page_size = page_size
        self.strategy = strategy
        self.capacity_tokens: Optional[int] = capacity_tokens
        self.mamba_state_token_equiv = mamba_state_token_equiv
        self.tree = RadixTree(mamba_state_token_equiv=mamba_state_token_equiv)
        self.state = SimulationState()

    def reset(self) -> None:
        self.tree = RadixTree(mamba_state_token_equiv=self.mamba_state_token_equiv)
        self.state = SimulationState()

    def process_token_ids(self, token_ids: List[int]) -> RequestTrace:
        pages = tokens_to_pages(token_ids, self.page_size)
        hit_pages, hit_tokens, kv_only_hit_pages, total_tokens, turn_hit_tokens, new_nodes = (
            self.tree.simulate_request(pages)
        )

        # Admission: for each newly inserted node, ask the strategy whether to
        # store a Mamba state.  Skipped entirely in pure full-attention mode.
        if self.mamba_state_token_equiv > 0:
            for node in new_nodes:
                if self.strategy.admit_mamba_state(node):
                    self.tree.set_mamba_state(node)
            self.strategy.on_new_nodes_inserted(self.tree, new_nodes)

        trace = RequestTrace(
            input_tokens=total_tokens,
            hit_tokens=hit_tokens,
            hit_pages=hit_pages,
            total_pages=len(pages),
            turn_hit_tokens=turn_hit_tokens,
            kv_only_hit_pages=kv_only_hit_pages,
        )
        self._evict_until_fit()
        self.state.record_trace(trace, self.tree.total_cached_tokens())
        return trace

    def _evict_until_fit(self) -> None:
        if self.capacity_tokens is None:
            return
        cap = self.capacity_tokens
        while self.tree.total_cached_tokens() > cap:
            # Try Mamba-state-only evictions first: cheaper to demote a node
            # (keep its KV pages, drop its SSM state) than to remove it entirely.
            mamba_candidates = self.strategy.select_mamba_state_evictions(self.tree, 1)
            if mamba_candidates:
                self.tree.evict_mamba_state(mamba_candidates[0])
                continue
            # Fall back to full node eviction.
            node_candidates = self.strategy.select_nodes(self.tree, 1)
            if not node_candidates:
                break
            self.tree.remove_leaf(node_candidates[0])


class MultiTierCacheSimulator:
    """Placeholder for a two-level host-memory cache (not implemented)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._args = args
        self._kwargs = kwargs

    def process_token_ids(self, token_ids: List[int]) -> RequestTrace:
        raise NotImplementedError(
            "Multi-tier cache simulation is reserved for future work; "
            "use KVCacheSimulator for single-tier experiments."
        )
