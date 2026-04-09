"""KV prefix cache simulator with radix tree and pluggable eviction / admission.

Single-tier mode (HBM only) is the default.  When ``dram_strategy`` and
``dram_capacity_tokens > 0`` are both supplied, the simulator runs as a
two-tier cache (HBM + DRAM).  The DRAM tier is a *superset* of HBM —
every node created in HBM is immediately mirrored to DRAM.

Per-request flow in two-tier mode:
  1. Match + insert in the HBM tree (primary).
  2. Read-only prefix match in the DRAM tree for additional depth
     (from nodes that HBM evicted previously but DRAM retains).
  3. Mirror the full page sequence to DRAM (+ sync mamba state from HBM;
     DRAM never computes its own mamba state).
  4. Effective hit = HBM hits + extra DRAM hits beyond HBM depth.
  5. Evict HBM excess (no demotion — DRAM already has it).
  6. Evict DRAM excess → discard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.model_config import ModelConfig
from src.radix_tree import PageKey, RadixNode, RadixTree
from src.strategies.base import (
    EvictionStrategy,
    PageStatus,
    RequestPlan,
)


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
    """Per-request statistics.

    ``hit_tokens`` / ``hit_pages`` are the *effective* totals (HBM + DRAM).
    The HBM-only and DRAM-extra portions are also recorded for tier breakdowns.
    DRAM-related fields are zero when DRAM is disabled.
    """

    input_tokens: int
    hit_tokens: int
    hit_pages: int
    total_pages: int
    turn_hit_tokens: int
    is_branch: bool = False
    is_new_branch: bool = False
    # Tier breakdown (DRAM fields are 0 in single-tier mode).
    hbm_hit_tokens: int = 0
    dram_hit_tokens: int = 0
    promoted_tokens: int = 0      # DRAM → HBM (counted as DRAM-extra contribution)
    promoted_nodes: int = 0
    demoted_tokens: int = 0       # HBM → DRAM (this request)
    demoted_nodes: int = 0

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
    dram_usage_samples: List[int] = field(default_factory=list)

    def record_trace(
        self,
        t: RequestTrace,
        cached_tokens_after: int,
        dram_cached_tokens_after: int = 0,
    ) -> None:
        self.traces.append(t)
        self.usage_samples.append(cached_tokens_after)
        self.dram_usage_samples.append(dram_cached_tokens_after)


class KVCacheSimulator:
    """Page-based radix cache simulator with optional DRAM tier.

    Parameters
    ----------
    page_size:
        Number of tokens per cache page.
    strategy:
        Eviction (and optionally admission) strategy for the HBM tier.
    capacity_tokens:
        Maximum effective token-equivalent units the HBM cache may hold.
        ``None`` = unlimited.  In hybrid mode this budget includes Mamba state
        overhead (each state counts as ``model.mamba_state_token_equiv`` tokens).
    dram_strategy / dram_capacity_tokens:
        DRAM tier eviction strategy and capacity.  DRAM is enabled iff
        ``dram_strategy is not None`` AND ``dram_capacity_tokens != 0``.
        ``dram_capacity_tokens=None`` means unlimited DRAM (no eviction);
        a positive int caps the tier; ``0`` (default) disables DRAM.
    model:
        Optional :class:`~src.model_config.ModelConfig` describing the
        target model.  Hybrid behaviour (Mamba SSM state accounting and
        hit gating) is enabled when the model carries SSM layers; pure
        transformers — or ``None`` — collapse to plain full-attention mode.
    gpu_flops / pcie_bandwidth:
        When supplied together with ``model``, the simulator activates
        **smart DRAM fallback**: a DRAM hit is *not* counted (and the
        suffix is recomputed instead) when its PCIe transfer time would
        exceed the compute it would save.  In that case the trace records
        ``dram_hit_tokens=0`` for that request, so ``dram_token_hit_rate``
        and ``pcie_total_transfer_*`` reflect only useful DRAM hits.
        When any of the three is missing, the simulator falls back to the
        original "always use every DRAM hit" policy.
    logger:
        Optional :class:`~viz.tree_logger.TreeLogger`.  Only the HBM tier is
        currently logged.
    """

    def __init__(
        self,
        page_size: int,
        strategy: EvictionStrategy,
        capacity_tokens: Optional[int] = None,
        dram_strategy: Optional[EvictionStrategy] = None,
        dram_capacity_tokens: Optional[int] = 0,
        model: Optional["ModelConfig"] = None,
        logger: object = None,
        *,
        gpu_flops: Optional[float] = None,
        pcie_bandwidth: Optional[float] = None,
    ) -> None:
        self.page_size = page_size
        self.strategy = strategy
        self.capacity_tokens: Optional[int] = capacity_tokens
        self.model = model
        self.gpu_flops = gpu_flops
        self.pcie_bandwidth = pcie_bandwidth
        self.logger = logger
        self.tree = RadixTree(model=model)
        # Smart DRAM fallback is only meaningful with full hardware info.
        self._smart_dram_fallback = (
            model is not None
            and gpu_flops is not None
            and gpu_flops > 0
            and pcie_bandwidth is not None
            and pcie_bandwidth > 0
        )

        # DRAM enabled when a strategy is supplied AND capacity != 0.
        # ``None`` means unlimited; a positive int means bounded.
        self.dram_enabled = dram_strategy is not None and dram_capacity_tokens != 0
        self.dram_strategy = dram_strategy if self.dram_enabled else None
        self.dram_capacity: Optional[int] = (
            dram_capacity_tokens if self.dram_enabled else 0
        )
        self.dram_tree: Optional[RadixTree] = (
            RadixTree(model=model) if self.dram_enabled else None
        )

        self.state = SimulationState()
        self._request_count = 0

    def reset(self) -> None:
        self.tree = RadixTree(model=self.model)
        if self.dram_enabled:
            self.dram_tree = RadixTree(model=self.model)
        self.state = SimulationState()

    def process_token_ids(self, token_ids: List[int]) -> RequestTrace:
        pages = tokens_to_pages(token_ids, self.page_size)
        # Always drop a trailing partial page — every strategy treats
        # them the same, so the simulator owns this normalisation.
        if len(pages) > 1 and len(pages[-1]) < self.page_size:
            pages = pages[:-1]
        total_tokens = sum(len(p) for p in pages)

        log = self.logger
        rid = self._request_count
        self._request_count += 1

        # ----------------------------------------------------------------
        # 1. HBM: walk + split (no insertion yet).
        # ----------------------------------------------------------------
        matched_nodes, page_idx = self.tree.match_and_split(pages)
        remaining_pages = pages[page_idx:]
        hit_pages, hit_tokens, turn_hit_tokens = self._compute_hit(
            self.tree, matched_nodes
        )

        # Branch detection (before the new leaf is attached).
        parent_for_insert = matched_nodes[-1] if matched_nodes else self.tree.root
        is_branch = bool(remaining_pages) and len(parent_for_insert.children) > 0
        is_new_branch = is_branch and not parent_for_insert.has_mamba_state

        # ----------------------------------------------------------------
        # 2. Strategy hooks: cache-hit bookkeeping + plan the request.
        # ----------------------------------------------------------------
        if matched_nodes:
            self.strategy.on_cache_hit(self.tree, matched_nodes)
        plan = self.strategy.plan_request(
            self.tree, matched_nodes, remaining_pages
        )
        assert len(plan.remaining) == len(remaining_pages), (
            "plan_request must return one PageStatus per remaining page"
        )

        # ----------------------------------------------------------------
        # 3. Apply the plan to HBM.
        # ----------------------------------------------------------------
        hbm_new_nodes, hbm_mamba_nodes, hbm_end_depth = self._apply_plan(
            self.tree, pages, matched_nodes, remaining_pages, plan
        )
        if hbm_new_nodes:
            self.strategy.on_nodes_inserted(self.tree, hbm_new_nodes)

        # ----------------------------------------------------------------
        # 4. DRAM tier (superset mirror).  DRAM is NOT a 1:1 op replay:
        # the simulator walks DRAM independently, computes its own hit
        # (mamba-gated), then copies HBM's final request-path state to
        # DRAM — any pages HBM ended up with plus the mamba positions
        # HBM just created.
        # ----------------------------------------------------------------
        dram_extra_tokens = 0
        dram_extra_pages = 0
        dram_new_nodes: List[RadixNode] = []
        if self.dram_enabled:
            assert self.dram_tree is not None and self.dram_strategy is not None

            dram_matched, dram_page_idx = self.dram_tree.match_and_split(pages)
            dram_hit_pages, dram_hit_tokens, _ = self._compute_hit(
                self.dram_tree, dram_matched
            )

            # DRAM extra hit = any mamba-gated depth DRAM matched beyond HBM.
            if dram_hit_tokens > hit_tokens:
                candidate_extra_tokens = dram_hit_tokens - hit_tokens
                candidate_extra_pages = dram_hit_pages - hit_pages
                use_dram = True
                if self._smart_dram_fallback:
                    assert (
                        self.model is not None
                        and self.gpu_flops is not None
                        and self.pcie_bandwidth is not None
                    )
                    flop_hbm_only = self.model.incremental_prefill_flop(
                        total_tokens, hit_tokens
                    )
                    flop_full_hit = self.model.incremental_prefill_flop(
                        total_tokens, hit_tokens + candidate_extra_tokens
                    )
                    compute_saved = (
                        flop_hbm_only - flop_full_hit
                    ) / self.gpu_flops
                    transfer_time = (
                        candidate_extra_tokens
                        * self.model.kv_bytes_per_token
                        / self.pcie_bandwidth
                    )
                    if transfer_time > compute_saved:
                        use_dram = False
                if use_dram:
                    dram_extra_tokens = candidate_extra_tokens
                    dram_extra_pages = candidate_extra_pages

            # DRAM cache-hit bookkeeping.  ``on_cache_hit`` is a pure
            # observer (see the contract in strategies/base.py) — it may
            # only touch strategy-owned external attributes, never the
            # tree's mamba state — so no snapshot/restore is needed.
            if dram_matched:
                self.dram_strategy.on_cache_hit(self.dram_tree, dram_matched)

            # Mirror HBM's request-path state to DRAM.
            dram_new_nodes = self._mirror_hbm_to_dram(
                pages, dram_matched, hbm_end_depth
            )

            if dram_new_nodes:
                self.dram_strategy.on_nodes_inserted(
                    self.dram_tree, dram_new_nodes
                )

        effective_hit_tokens = hit_tokens + dram_extra_tokens
        effective_hit_pages = hit_pages + dram_extra_pages

        # ----------------------------------------------------------------
        # 5. Logging (after all mutations have settled).
        # ----------------------------------------------------------------
        pending_splits = self.tree.drain_pending_splits()
        if self.dram_tree is not None:
            self.dram_tree.drain_pending_splits()
        if log:
            log.request_start(rid, self.tree.clock, total_tokens)
            for sp in pending_splits:
                log.split(*sp)
            for n in matched_nodes:
                log.hit(n)
            for n in hbm_new_nodes:
                log.insert(n)
            for n in hbm_mamba_nodes:
                log.mamba_set(n)

        # ----------------------------------------------------------------
        # 6. Evict HBM excess, then DRAM excess.
        # ----------------------------------------------------------------
        demoted_tokens, demoted_nodes = self._evict_until_fit()

        trace = RequestTrace(
            input_tokens=total_tokens,
            hit_tokens=effective_hit_tokens,
            hit_pages=effective_hit_pages,
            total_pages=len(pages),
            turn_hit_tokens=turn_hit_tokens,
            is_branch=is_branch,
            is_new_branch=is_new_branch,
            hbm_hit_tokens=hit_tokens,
            dram_hit_tokens=dram_extra_tokens,
            promoted_tokens=dram_extra_tokens,
            promoted_nodes=dram_extra_pages,
            demoted_tokens=demoted_tokens,
            demoted_nodes=demoted_nodes,
        )

        cached = self.tree.total_cached_tokens()
        if log:
            log.request_end(
                rid, effective_hit_tokens, total_tokens - effective_hit_tokens, cached
            )

        dram_cached = (
            self.dram_tree.total_cached_tokens()
            if self.dram_enabled and self.dram_tree
            else 0
        )
        self.state.record_trace(trace, cached, dram_cached)
        return trace

    def _compute_hit(
        self, tree: RadixTree, matched_nodes: List[RadixNode]
    ) -> Tuple[int, int, int]:
        """Effective hit (mamba-gated in hybrid mode) + turn-hit tokens.

        The hit extends from the root to the deepest matched node that
        carries a mamba state.  In pure-transformer mode (no SSM layers)
        every matched node counts toward the hit.

        Returns ``(hit_pages, hit_tokens, turn_hit_tokens)``.
        """
        if not matched_nodes:
            return 0, 0, 0

        cum_pages: List[int] = []
        cum_tokens: List[int] = []
        cp = ct = 0
        for n in matched_nodes:
            cp += n.num_pages
            ct += n.num_tokens
            cum_pages.append(cp)
            cum_tokens.append(ct)

        if tree.mamba_state_token_equiv > 0:
            effective_hit_idx = -1
            for i, n in enumerate(matched_nodes):
                if n.has_mamba_state:
                    effective_hit_idx = i
            if effective_hit_idx >= 0:
                hit_pages = cum_pages[effective_hit_idx]
                hit_tokens = cum_tokens[effective_hit_idx]
            else:
                hit_pages = 0
                hit_tokens = 0
            effective_range = effective_hit_idx + 1
        else:
            hit_pages = cp
            hit_tokens = ct
            effective_range = len(matched_nodes)

        turn_hit_tokens = 0
        for i in range(effective_range):
            if matched_nodes[i].is_turn_end:
                turn_hit_tokens = cum_tokens[i]

        return hit_pages, hit_tokens, turn_hit_tokens

    def _apply_plan(
        self,
        tree: RadixTree,
        pages: List[PageKey],
        matched_nodes: List[RadixNode],
        remaining_pages: List[PageKey],
        plan: RequestPlan,
    ) -> Tuple[List[RadixNode], List[RadixNode], int]:
        """Apply *plan* to HBM (or any tree with a local match).

        1. Walk ``plan.remaining`` and find the longest prefix that is
           not ``IGNORE`` — this is the admitted suffix.
        2. Insert it as a single compressed leaf.
        3. For each ``KV_AND_MAMBA`` page, place mamba state at the
           boundary at the end of that page.
        4. If ``plan.mamba_on_matched_parent`` is True, also set mamba
           on the deepest matched node.

        Returns ``(new_nodes, mamba_nodes, end_depth)`` where ``end_depth``
        is the final depth (tokens from root) of the node chain after
        applying the plan.  ``end_depth`` equals the matched depth when
        no pages were admitted.
        """
        new_nodes: List[RadixNode] = []
        mamba_nodes: List[RadixNode] = []

        parent: RadixNode = matched_nodes[-1] if matched_nodes else tree.root
        matched_depth = parent.depth_tokens

        # Matched-parent mamba (Marconi fork-point style).
        if (
            plan.mamba_on_matched_parent
            and parent is not tree.root
            and not parent.has_mamba_state
        ):
            tree.set_mamba_state(parent)
            mamba_nodes.append(parent)

        # Admitted prefix = longest run before the first IGNORE.
        admit_count = len(plan.remaining)
        for i, status in enumerate(plan.remaining):
            if status == PageStatus.IGNORE:
                admit_count = i
                break

        if admit_count == 0:
            return new_nodes, mamba_nodes, matched_depth

        admit_pages = remaining_pages[:admit_count]
        leaf = tree.insert_leaf_at(parent, tuple(admit_pages))
        new_nodes.append(leaf)
        end_depth = leaf.depth_tokens

        # Place mamba at the end of each KV_AND_MAMBA page within the
        # admitted prefix.  set_mamba_at_depth snaps to the correct
        # page boundary and splits the leaf as needed.
        cum = 0
        for i in range(admit_count):
            cum += len(admit_pages[i])
            if plan.remaining[i] == PageStatus.KV_AND_MAMBA:
                depth = matched_depth + cum
                target = tree.set_mamba_at_depth(pages, depth)
                if target is not None:
                    mamba_nodes.append(target)

        return new_nodes, mamba_nodes, end_depth

    def _mirror_hbm_to_dram(
        self,
        pages: List[PageKey],
        dram_matched: List[RadixNode],
        hbm_end_depth: int,
    ) -> List[RadixNode]:
        """Copy HBM's post-request state onto the DRAM tree.

        This is NOT a 1:1 op replay of the plan.  The simulator instead
        mirrors HBM's actual request-path state by:

        1. Ensuring DRAM has every page HBM has along the request path
           up to ``hbm_end_depth`` (insert a leaf when DRAM's own match
           is shallower than HBM's end).
        2. Walking HBM along the request's page sequence and, for every
           HBM node that currently carries a mamba state, calling
           ``dram_tree.set_mamba_at_depth`` at the same absolute depth.
           ``set_mamba_at_depth`` is idempotent, so mamba positions
           inherited from earlier requests are harmless no-ops.
        """
        assert self.dram_tree is not None

        new_nodes: List[RadixNode] = []

        dram_parent: RadixNode = (
            dram_matched[-1] if dram_matched else self.dram_tree.root
        )
        dram_depth = dram_parent.depth_tokens

        # (1) Fill the page gap if DRAM is shallower than HBM's end.
        if hbm_end_depth > dram_depth:
            tokens_needed = hbm_end_depth - dram_depth
            dram_consumed_pages = sum(n.num_pages for n in dram_matched)
            to_insert: List[PageKey] = []
            tokens_added = 0
            for p in pages[dram_consumed_pages:]:
                if tokens_added >= tokens_needed:
                    break
                to_insert.append(p)
                tokens_added += len(p)

            if to_insert:
                leaf = self.dram_tree.insert_leaf_at(
                    dram_parent, tuple(to_insert)
                )
                new_nodes.append(leaf)

        # (2) Walk HBM along the request path; sync every mamba-state
        # depth to DRAM.  This is robust to splits, to duplicate depths
        # across subtrees, and to previously-synced mamba state.
        hbm_node = self.tree.root
        page_idx = 0
        while page_idx < len(pages):
            ch = hbm_node.children.get(pages[page_idx])
            if ch is None:
                break
            if ch.has_mamba_state:
                self.dram_tree.set_mamba_at_depth(pages, ch.depth_tokens)
            hbm_node = ch
            page_idx += ch.num_pages

        return new_nodes


    def _evict_until_fit(self) -> Tuple[int, int]:
        """Evict HBM, then evict DRAM.

        DRAM is already a superset (mirrored on every request), so HBM
        eviction simply removes nodes — no demotion step.

        Returns (demoted_tokens, demoted_nodes) for trace accounting.
        """
        demoted_tokens = 0
        demoted_nodes = 0

        if self.capacity_tokens is not None:
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
                    continue

                removed = self.tree.remove_leaf(node)
                if self.dram_enabled:
                    for rn in removed:
                        demoted_tokens += rn.num_tokens
                        demoted_nodes += 1
                if log and removed:
                    log.evict(removed)

        if self.dram_enabled and self.dram_capacity is not None:
            assert self.dram_tree is not None and self.dram_strategy is not None
            while self.dram_tree.total_cached_tokens() > self.dram_capacity:
                action = self.dram_strategy.select_eviction(self.dram_tree)
                if action is None:
                    break
                victim, op = action
                if op == "mamba":
                    self.dram_tree.evict_mamba_state(victim)
                else:
                    self.dram_tree.remove_leaf(victim)

        return demoted_tokens, demoted_nodes
