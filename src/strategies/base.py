"""Abstract eviction / admission strategy interface.

A strategy owns three decisions:

1. **Eviction** — which cached pages (tree nodes) to evict, via
   :meth:`EvictionStrategy.select_eviction`.

2. **Per-request admission + mamba placement**, via
   :meth:`EvictionStrategy.plan_request`.  The strategy receives the
   matched prefix and the list of ``remaining_pages`` the request
   would like to cache, and returns a :class:`RequestPlan` describing:

   * a :class:`PageStatus` for *each* remaining page
     (``IGNORE`` / ``KV_ONLY`` / ``KV_AND_MAMBA``), and
   * whether to set mamba state on the deepest matched node (the
     "fork-point parent" pattern from Marconi).

   ``IGNORE`` is sticky: once the status flips to ``IGNORE`` at page
   ``i`` the simulator stops admitting and drops everything from ``i``
   onward — you can't have a hole in the middle of a prefix.

3. **Per-node bookkeeping** on cache hits and after insertions, via
   :meth:`on_cache_hit` and :meth:`on_nodes_inserted`.  These are
   passive observers — they MUST NOT call ``tree.set_mamba_state``
   or otherwise mutate the cache's mamba layout.

Mirroring to DRAM
-----------------
The simulator calls ``plan_request`` **only** for HBM.  DRAM is then
mirrored from HBM's post-request state: every page HBM ended up
caching is copied to DRAM, along with every mamba position HBM newly
created.  DRAM's own eviction strategy still runs, and bookkeeping
hooks are invoked on DRAM so online metadata (LRU timestamps, CRF
scores) can track the tier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

from src.radix_tree import PageKey

if TYPE_CHECKING:
    from src.model_config import ModelConfig
    from src.radix_tree import RadixNode, RadixTree

EvictOp = Literal["mamba", "leaf"]


# ---------------------------------------------------------------------------
# Per-page admission decision
# ---------------------------------------------------------------------------


class PageStatus(Enum):
    """What the strategy wants done with one remaining page of a request.

    The simulator interprets the list of statuses from left to right:

    * ``IGNORE``       — do not cache this page.  Any page at or after
                         the first ``IGNORE`` is also dropped.
    * ``KV_ONLY``      — admit the page as KV cache.
    * ``KV_AND_MAMBA`` — admit the page as KV cache AND place a mamba
                         state at the boundary at the end of this page.
    """

    IGNORE = 0
    KV_ONLY = 1
    KV_AND_MAMBA = 2


@dataclass
class RequestPlan:
    """Strategy-decided plan for one request.

    Parameters
    ----------
    remaining:
        One :class:`PageStatus` per page in ``remaining_pages``
        (the simulator asserts ``len(remaining) == len(remaining_pages)``).
    mamba_on_matched_parent:
        If ``True``, also set mamba state on the deepest matched node
        (Marconi's fork-point parent pattern).  Ignored when there are
        no matched nodes or the deepest matched node already has mamba.
    """

    remaining: List[PageStatus] = field(default_factory=list)
    mamba_on_matched_parent: bool = False


# ---------------------------------------------------------------------------
# Hardware-aware mamba admission threshold
# ---------------------------------------------------------------------------


def compute_min_mamba_admit_depth(
    model: "ModelConfig",
    gpu_flops: Optional[float],
    pcie_bandwidth: Optional[float],
) -> int:
    """Smallest prefix depth (in tokens) at which storing a Mamba state is
    worth its hardware cost.

    A Mamba state is "worth keeping" only if loading it via PCIe would be
    cheaper than recomputing the prefix from scratch:

        prefill_flop(depth) / gpu_flops  >  mamba_state_bytes / pcie_bandwidth

    The function returns the smallest ``depth`` satisfying that
    inequality, found via exponential + binary search.  Returns ``0``
    (admit anything) when:
      * the model has no SSM layers (no Mamba state to begin with),
      * either hardware throughput is missing or non-positive (no
        meaningful threshold can be derived).
    """
    if model.num_ssm_layers == 0:
        return 0
    mamba_bytes = model.mamba_state_bytes_total
    if mamba_bytes <= 0:
        return 0
    if gpu_flops is None or gpu_flops <= 0:
        return 0
    if pcie_bandwidth is None or pcie_bandwidth <= 0:
        return 0

    target_flop = (mamba_bytes / pcie_bandwidth) * gpu_flops
    if target_flop <= 0:
        return 0
    # Exponential search for an upper bound where prefill_flop >= target.
    hi = 1
    while model.prefill_flop(hi) < target_flop:
        hi *= 2
        if hi > 10_000_000:
            return hi  # Effectively never admit; cap to avoid runaway.
    # Binary search the smallest depth meeting the bound.
    lo = max(1, hi // 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if model.prefill_flop(mid) >= target_flop:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class EvictionStrategy(ABC):
    """Decides cache admission, mamba-state placement, and eviction.

    Subclasses must implement :meth:`select_eviction`.  All other hooks
    have sensible defaults.
    """

    @abstractmethod
    def select_eviction(
        self, tree: "RadixTree"
    ) -> Optional[Tuple["RadixNode", EvictOp]]:
        """Pick the single best eviction action.

        Returns ``(node, "mamba")`` to drop only the Mamba state, or
        ``(node, "leaf")`` to remove the leaf entirely, or ``None``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Per-request planning
    # ------------------------------------------------------------------

    def plan_request(
        self,
        tree: "RadixTree",
        matched_nodes: List["RadixNode"],
        remaining_pages: List[PageKey],
    ) -> RequestPlan:
        """Decide what to do with each remaining page of a request.

        Default behaviour: admit every remaining page; in hybrid mode,
        flip the last page to ``KV_AND_MAMBA`` so the request's turn
        boundary is checkpointed.  No matched-node mamba placement.
        """
        if not remaining_pages:
            return RequestPlan(remaining=[])

        default_status = PageStatus.KV_ONLY
        statuses = [default_status] * len(remaining_pages)
        if tree.mamba_state_token_equiv > 0:
            statuses[-1] = PageStatus.KV_AND_MAMBA
        return RequestPlan(remaining=statuses)

    # ------------------------------------------------------------------
    # Bookkeeping hooks (pure observers of cache state)
    # ------------------------------------------------------------------

    def on_cache_hit(
        self, tree: "RadixTree", matched_nodes: List["RadixNode"]
    ) -> None:
        """Called once per request after the prefix match.

        **Strict contract** — this hook is a read-only observer.  It
        **MUST NOT** mutate any tree-managed node state, including but
        not limited to: ``node.pages``, ``node.children``, ``num_tokens``,
        ``depth_tokens``, ``is_turn_end``, ``has_mamba_state``.  Do not
        call ``tree.set_mamba_state`` or ``tree.evict_mamba_state``
        from this hook.

        Strategies may still write **their own** per-node metadata via
        ad-hoc attributes (``node._branch_lru``, ``node._crf_value``,
        etc.) or in external dicts.  Mamba and KV admission decisions
        belong in :meth:`plan_request`.  The simulator relies on this
        invariant: DRAM's ``on_cache_hit`` is called without any
        snapshot/restore around mamba state, so a hook that cheats
        will silently diverge the tiers.
        """
        return

    def on_nodes_inserted(
        self, tree: "RadixTree", new_nodes: List["RadixNode"]
    ) -> None:
        """Called once per request after the strategy's plan has been
        applied to the tree, with the list of newly created nodes.

        Same strict contract as :meth:`on_cache_hit`: read-only with
        respect to tree-managed node state.  Use for initialising
        strategy-owned external metadata (CRF / LRU timestamps) only.
        Default: no-op.
        """
        return
