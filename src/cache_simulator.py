"""KV prefix cache simulator with radix tree and pluggable eviction / admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.radix_tree import PageKey, RadixTree, node_page_path
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
    is_branch: bool = False
    is_new_branch: bool = False

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
        hit_pages, hit_tokens, total_tokens, turn_hit_tokens, new_nodes, matched_nodes, is_branch, is_new_branch = (
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
            is_branch=is_branch,
            is_new_branch=is_new_branch,
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


@dataclass
class MultiTierRequestTrace:
    """Per-request statistics for two-tier (HBM + DRAM) cache."""

    input_tokens: int
    hit_tokens: int          # effective total = hbm + dram hits
    hit_pages: int
    total_pages: int
    turn_hit_tokens: int
    hbm_hit_tokens: int
    dram_hit_tokens: int
    promoted_tokens: int     # DRAM → HBM this request
    promoted_nodes: int
    demoted_tokens: int      # HBM → DRAM this request
    demoted_nodes: int
    is_branch: bool = False
    is_new_branch: bool = False

    @property
    def miss_tokens(self) -> int:
        return self.input_tokens - self.hit_tokens

    @property
    def per_request_token_hit_rate(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return self.hit_tokens / self.input_tokens


@dataclass
class MultiTierSimulationState:
    """Accumulated counters for a multi-tier simulation run."""

    traces: List[MultiTierRequestTrace] = field(default_factory=list)
    hbm_usage_samples: List[int] = field(default_factory=list)
    dram_usage_samples: List[int] = field(default_factory=list)

    def record_trace(
        self,
        t: MultiTierRequestTrace,
        hbm_cached_after: int,
        dram_cached_after: int,
    ) -> None:
        self.traces.append(t)
        self.hbm_usage_samples.append(hbm_cached_after)
        self.dram_usage_samples.append(dram_cached_after)


class MultiTierCacheSimulator:
    """Two-tier prefix KV cache: HBM (fast/small) + DRAM (slow/large).

    Each tier maintains an independent :class:`RadixTree` with its own
    eviction strategy and capacity.  The DRAM tier is **inclusive** — it
    may contain prefix paths that also exist in HBM, which keeps the
    implementation simple (each tree is self-contained).

    Lookup flow per request:
      1. Match + insert in the HBM tree (primary).
      2. Read-only prefix match in the DRAM tree for additional depth.
      3. Effective hit = HBM hits + extra DRAM hits beyond HBM depth.
      4. Evict HBM excess → demote evicted leaf paths to DRAM.
      5. Evict DRAM excess → discard.

    Parameters
    ----------
    page_size:
        Tokens per cache page.
    hbm_strategy / dram_strategy:
        Eviction strategies for each tier.
    hbm_capacity_tokens / dram_capacity_tokens:
        Maximum token-equivalent capacity per tier.
    """

    def __init__(
        self,
        page_size: int,
        hbm_strategy: EvictionStrategy,
        dram_strategy: EvictionStrategy,
        hbm_capacity_tokens: int,
        dram_capacity_tokens: int,
        mamba_state_token_equiv: int = 0,
    ) -> None:
        self.page_size = page_size
        self.hbm_strategy = hbm_strategy
        self.dram_strategy = dram_strategy
        self.hbm_capacity = hbm_capacity_tokens
        self.dram_capacity = dram_capacity_tokens
        self.mamba_state_token_equiv = mamba_state_token_equiv

        self.hbm_tree = RadixTree(mamba_state_token_equiv=mamba_state_token_equiv)
        self.dram_tree = RadixTree(mamba_state_token_equiv=mamba_state_token_equiv)
        self.state = MultiTierSimulationState()
        self._request_count = 0

    # -- public API -------------------------------------------------------

    def process_token_ids(self, token_ids: List[int]) -> MultiTierRequestTrace:
        pages = tokens_to_pages(token_ids, self.page_size)
        if (
            self.hbm_strategy.drop_partial_last_page
            and len(pages) > 1
            and len(pages[-1]) < self.page_size
        ):
            pages = pages[:-1]

        self._request_count += 1

        # ── Step 1: HBM lookup + insert suffix ──────────────────────
        hbm_hit_pages, hbm_hit_tokens, total_tokens, turn_hit_tokens, hbm_new, hbm_matched, _is_branch, _is_new_branch = (
            self.hbm_tree.simulate_request(pages)
        )
        if hbm_matched:
            self.hbm_strategy.on_cache_hit(self.hbm_tree, hbm_matched)
        if hbm_new:
            self.hbm_strategy.on_new_nodes_inserted(self.hbm_tree, hbm_new)

        # ── Step 2: DRAM read-only lookup for extra depth ───────────
        dram_extra_tokens = 0
        dram_extra_pages = 0
        if hbm_hit_pages < len(pages):
            dram_match_pages, dram_match_tokens, dram_matched = (
                self.dram_tree.prefix_match(pages)
            )
            if dram_match_pages > hbm_hit_pages:
                dram_extra_pages = dram_match_pages - hbm_hit_pages
                dram_extra_tokens = dram_match_tokens - hbm_hit_tokens
            # Touch DRAM matched nodes so recency info stays current.
            if dram_matched:
                self.dram_tree._clock += 1
                ts = self.dram_tree._clock
                for n in dram_matched:
                    n.touch(ts)
                self.dram_strategy.on_cache_hit(self.dram_tree, dram_matched)

        effective_hit_tokens = hbm_hit_tokens + dram_extra_tokens
        effective_hit_pages = hbm_hit_pages + dram_extra_pages

        # ── Step 3: Evict HBM → demote to DRAM ─────────────────────
        demoted_tokens = 0
        demoted_nodes = 0
        while self.hbm_tree.total_cached_tokens() > self.hbm_capacity:
            action = self.hbm_strategy.select_eviction(self.hbm_tree)
            if action is None:
                break
            victim, op = action
            if op == "mamba":
                self.hbm_tree.evict_mamba_state(victim)
            else:
                # Reconstruct full page path and demote to DRAM.
                # This must happen BEFORE remove_leaf, while parent pointers
                # are still valid.
                full_pages = node_page_path(victim)
                if full_pages:
                    dram_new = self.dram_tree.insert_pages(
                        full_pages,
                        timestamp=victim.last_access,
                        access_count=victim.access_count,
                    )
                    if dram_new:
                        self.dram_strategy.on_new_nodes_inserted(
                            self.dram_tree, dram_new
                        )
                # remove_leaf returns victim + pruned ancestors.
                removed = self.hbm_tree.remove_leaf(victim)
                for rn in removed:
                    demoted_tokens += rn.num_tokens
                    demoted_nodes += 1

        # ── Step 4: Evict DRAM → discard ────────────────────────────
        while self.dram_tree.total_cached_tokens() > self.dram_capacity:
            action = self.dram_strategy.select_eviction(self.dram_tree)
            if action is None:
                break
            victim, op = action
            if op == "mamba":
                self.dram_tree.evict_mamba_state(victim)
            else:
                self.dram_tree.remove_leaf(victim)

        # ── Step 5: Record trace ────────────────────────────────────
        # Promotion: DRAM extra tokens are "promoted" in the sense that
        # they were found in DRAM and are now also in HBM (inserted by
        # simulate_request in step 1).  We count the DRAM-exclusive
        # contribution as promoted volume.
        promoted_tokens = dram_extra_tokens
        promoted_nodes = dram_extra_pages  # one node per extra matched page-group

        trace = MultiTierRequestTrace(
            input_tokens=total_tokens,
            hit_tokens=effective_hit_tokens,
            hit_pages=effective_hit_pages,
            total_pages=len(pages),
            turn_hit_tokens=turn_hit_tokens,
            hbm_hit_tokens=hbm_hit_tokens,
            dram_hit_tokens=dram_extra_tokens,
            promoted_tokens=promoted_tokens,
            promoted_nodes=promoted_nodes,
            demoted_tokens=demoted_tokens,
            demoted_nodes=demoted_nodes,
            is_branch=_is_branch,
            is_new_branch=_is_new_branch,
        )
        self.state.record_trace(
            trace,
            self.hbm_tree.total_cached_tokens(),
            self.dram_tree.total_cached_tokens(),
        )
        return trace
