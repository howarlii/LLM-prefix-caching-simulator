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
    logger:
        Optional :class:`~viz.tree_logger.TreeLogger`.  Passed through to the
        :class:`RadixTree` which handles all event recording.
    """

    def __init__(
        self,
        page_size: int,
        strategy: EvictionStrategy,
        capacity_tokens: Optional[int] = None,
        mamba_state_token_equiv: int = 0,
        logger: object = None,
    ) -> None:
        self.page_size = page_size
        self.strategy = strategy
        self.capacity_tokens: Optional[int] = capacity_tokens
        self.mamba_state_token_equiv = mamba_state_token_equiv
        self.logger = logger
        self.tree = RadixTree(mamba_state_token_equiv=mamba_state_token_equiv)
        self.state = SimulationState()
        self._request_count = 0

    def reset(self) -> None:
        self.tree = RadixTree(mamba_state_token_equiv=self.mamba_state_token_equiv)
        self.state = SimulationState()

    def process_token_ids(self, token_ids: List[int]) -> RequestTrace:
        pages = tokens_to_pages(token_ids, self.page_size)
        # Drop trailing partial page when the strategy requires full pages only.
        if (
            self.strategy.drop_partial_last_page
            and len(pages) > 1
            and len(pages[-1]) < self.page_size
        ):
            pages = pages[:-1]
        hit_pages, hit_tokens, total_tokens, turn_hit_tokens, new_nodes, matched_nodes = (
            self.tree.simulate_request(pages)
        )

        log = self.logger
        rid = self._request_count
        self._request_count += 1

        # 1. Notify strategy about cache hits (updates CRF etc.)
        if matched_nodes:
            self.strategy.on_cache_hit(self.tree, matched_nodes)

        # 2. Admission: in hybrid mode, ask strategy whether to store Mamba state.
        #    Track which nodes got mamba state for logging.
        mamba_admitted: List = []
        if self.mamba_state_token_equiv > 0:
            for node in new_nodes:
                if self.strategy.admit_mamba_state(node):
                    self.tree.set_mamba_state(node)
                    mamba_admitted.append(node)

        # 3. Notify strategy about new nodes (CRF init, fork detection, etc.).
        #    This may also set mamba state (e.g. Marconi fork-point parent).
        mamba_before = {id(n) for n in self.tree.nodes_with_mamba_state()} if log and new_nodes else set()
        if new_nodes:
            self.strategy.on_new_nodes_inserted(self.tree, new_nodes)

        # 4. Log AFTER all strategy hooks, so CRF values are up-to-date.
        # Always drain pending splits to avoid memory leak.
        pending_splits = self.tree.drain_pending_splits()
        if log:
            log.request_start(rid, self.tree.clock, total_tokens)
            # Splits (from simulate_request prefix walk + on_new_nodes_inserted)
            for sp in pending_splits:
                log.split(*sp)
            # Hits — CRF has been updated by on_cache_hit already
            for n in matched_nodes:
                log.hit(n)
            # Inserts
            for n in new_nodes:
                log.insert(n)
            # Mamba admissions from step 2
            for n in mamba_admitted:
                log.mamba_set(n)
            # Mamba admissions from step 3 (on_new_nodes_inserted)
            if new_nodes:
                for n in self.tree.nodes_with_mamba_state():
                    if id(n) not in mamba_before and n not in mamba_admitted:
                        log.mamba_set(n)

        trace = RequestTrace(
            input_tokens=total_tokens,
            hit_tokens=hit_tokens,
            hit_pages=hit_pages,
            total_pages=len(pages),
            turn_hit_tokens=turn_hit_tokens,
        )
        self._evict_until_fit()

        cached = self.tree.total_cached_tokens()
        if log:
            log.request_end(rid, hit_tokens, total_tokens - hit_tokens, cached)

        self.state.record_trace(trace, cached)
        return trace

    def _evict_until_fit(self) -> None:
        if self.capacity_tokens is None:
            return
        log = self.logger
        cap = self.capacity_tokens
        while self.tree.total_cached_tokens() > cap:
            action = self.strategy.select_eviction(self.tree)
            if action is None:
                break
            node, op = action
            if op == "mamba":
                self.tree.evict_mamba_state(node)
                if log:
                    log.mamba_evict(node)
            else:
                removed = self.tree.remove_leaf(node)
                if log and removed:
                    log.evict(removed)


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
