"""KV prefix cache simulator with radix tree and pluggable eviction."""

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
    """Processes tokenized requests through a page-based radix cache."""

    def __init__(
        self,
        page_size: int,
        strategy: EvictionStrategy,
        capacity_tokens: Optional[int] = None,
    ) -> None:
        self.page_size = page_size
        self.strategy = strategy
        self.capacity_tokens: Optional[int] = capacity_tokens
        self.tree = RadixTree()
        self.state = SimulationState()

    def reset(self) -> None:
        self.tree = RadixTree()
        self.state = SimulationState()

    def process_token_ids(self, token_ids: List[int]) -> RequestTrace:
        pages = tokens_to_pages(token_ids, self.page_size)
        hit_pages, hit_tokens, total_tokens = self.tree.simulate_request(pages)
        trace = RequestTrace(
            input_tokens=total_tokens,
            hit_tokens=hit_tokens,
            hit_pages=hit_pages,
            total_pages=len(pages),
        )
        self._evict_until_fit()
        self.state.record_trace(trace, self.tree.total_cached_tokens())
        return trace

    def _evict_until_fit(self) -> None:
        if self.capacity_tokens is None:
            return
        cap = self.capacity_tokens
        while self.tree.total_cached_tokens() > cap:
            # One leaf at a time so pruning stays consistent and strategies stay valid.
            candidates = self.strategy.select_nodes(self.tree, 1)
            if not candidates:
                break
            self.tree.remove_leaf(candidates[0])


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
