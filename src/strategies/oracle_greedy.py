"""Clairvoyant (offline-optimal) Bélády-style oracle strategy.

Given the **full** request sequence ahead of time, this strategy makes
eviction and Mamba-state admission decisions using a Bélády-style rule:
"evict the candidate whose next future use is the latest (or never)".

It is intended as an upper-bound benchmark — not as a deployable
strategy.  The greedy decisions are not provably optimal (the problem is
NP-hard with mamba-state placement, two tiers, and tree dependencies),
but in practice this is very close to the true optimum and trivially
beats every online strategy.

How the future is queried
-------------------------
At construction time the oracle:

1. Page-tokenises every request (using the same ``tokens_to_pages``
   helper as the simulator).
2. Builds a *global* radix tree by feeding all requests through
   ``RadixTree.simulate_request`` in order, with no eviction.
3. Walks each request through the **final** global tree to record, for
   every global node, the sorted list of request indices visiting it.
4. Indexes every prefix-depth on every multi-page node by its **per-page
   chain hash**, mapping the hash to the global node containing that
   prefix.

At simulation time, for any node in the simulator tree:

* Its per-page chain hash from root to its end is computed (cached on
  the node, invalidated automatically via ``id(self.pages)`` when the
  node is split).
* The hash is looked up in ``path_hash_to_global`` to find the
  corresponding global node.
* The global node's future-access list is bisect-searched for the
  smallest request index strictly greater than the current request — the
  "next use" used by Bélády.

Eviction rule
-------------
Among all eviction candidates (leaves and mamba-state internal nodes),
pick the one whose next use is *latest* in the future (or never).
Ties are broken by largest capacity cost (free more space first).

Mamba-state admission rule
--------------------------
For each new node on the current request's path:

* If the node is too shallow (loading the state via PCIe would be
  more expensive than recomputing from scratch — see
  ``compute_min_mamba_admit_depth``), refuse.
* Otherwise, admit only if at least one **future** request will visit
  this node (i.e., it might be a checkpoint).

Two-tier behaviour
------------------
The strategy does not need any special two-tier handling: the simulator
itself owns the HBM/DRAM cascade, and the same oracle instance can be
used for both tiers (a separate instance per tier is recommended so each
tier can track its own ``select_eviction`` calls independently).
"""

from __future__ import annotations

import bisect
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

from src.model_config import DEFAULT_MODEL, ModelConfig
from src.radix_tree import PageKey, RadixNode, RadixTree
from src.strategies.base import (
    EvictOp,
    EvictionStrategy,
    PageStatus,
    RequestPlan,
    compute_min_mamba_admit_depth,
)


def _tokens_to_pages(tokens: List[int], page_size: int) -> List[PageKey]:
    """Local copy of cache_simulator.tokens_to_pages to avoid a circular import."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    pages: List[PageKey] = []
    for i in range(0, len(tokens), page_size):
        pages.append(tuple(tokens[i : i + page_size]))
    return pages


class OracleGreedyStrategy(EvictionStrategy):
    """Clairvoyant Bélády-style oracle.

    Parameters
    ----------
    requests_token_ids:
        Full ordered list of every request's token ids — exactly the
        sequence that will be fed to the simulator.
    page_size:
        Cache page size (must match the simulator's ``page_size``).
    model:
        Model architecture (drives FLOP costs and ``mamba_state_token_equiv``).
    gpu_flops, pcie_bandwidth:
        Hardware throughput, used for the saved-time scoring and the
        depth-threshold for mamba-state admission.
    """

    def __init__(
        self,
        requests_token_ids: Sequence[Sequence[int]],
        page_size: int,
        model: ModelConfig = DEFAULT_MODEL,
        *,
        gpu_flops: Optional[float] = None,
        pcie_bandwidth: Optional[float] = None,
    ) -> None:
        self.model = model
        self.page_size = page_size
        self.gpu_flops = gpu_flops
        self.pcie_bandwidth = pcie_bandwidth
        self._min_mamba_admit_depth = compute_min_mamba_admit_depth(
            model, gpu_flops, pcie_bandwidth
        )

        # ----- Page-tokenise every request -----
        # The simulator always drops a trailing partial page, so we mirror
        # that here to keep the global tree's structure aligned.
        self.req_pages: List[List[PageKey]] = []
        for tids in requests_token_ids:
            pgs = _tokens_to_pages(list(tids), page_size)
            if len(pgs) > 1 and len(pgs[-1]) < page_size:
                pgs = pgs[:-1]
            self.req_pages.append(pgs)
        self.num_requests = len(self.req_pages)
        self.req_total_tokens: List[int] = [
            sum(len(p) for p in pgs) for pgs in self.req_pages
        ]

        # ----- Build global radix tree from all requests -----
        self.global_tree = RadixTree(model=model)
        for pgs in self.req_pages:
            self.global_tree.simulate_request(pgs)

        # ----- Index every intra-node prefix by per-page chain hash -----
        # Maps hash(prefix) → (global_node, intra_node_depth_tokens).
        # The prefix is the chain of page keys from root down to that depth.
        self.path_hash_to_global: Dict[int, RadixNode] = {}
        q: deque = deque([(self.global_tree.root, 0)])
        while q:
            n, parent_h = q.popleft()
            if n is self.global_tree.root:
                cur_h = 0
            else:
                cur_h = parent_h
                for p in n.pages:
                    cur_h = hash((cur_h, p))
                    # Every intra-node prefix maps to this same global node;
                    # the simulator may compress less aggressively, so we
                    # index every page boundary inside the compressed node.
                    self.path_hash_to_global[cur_h] = n
            for c in n.children.values():
                q.append((c, cur_h))

        # ----- For each global node, sorted list of visiting request idx -----
        # Walk each request through the *final* global tree (after all
        # splits have been applied) so the visit list reflects the
        # post-split topology.
        self.global_future_reqs: Dict[int, List[int]] = {}
        for ri, pgs in enumerate(self.req_pages):
            node = self.global_tree.root
            pi = 0
            while pi < len(pgs):
                ch = node.children.get(pgs[pi])
                if ch is None:
                    break
                # Verify exact match (the global tree built from these
                # exact requests should always match page-by-page).
                ok = True
                for j in range(len(ch.pages)):
                    if pi + j >= len(pgs) or pgs[pi + j] != ch.pages[j]:
                        ok = False
                        break
                if not ok:
                    break
                self.global_future_reqs.setdefault(id(ch), []).append(ri)
                pi += len(ch.pages)
                node = ch

    # ------------------------------------------------------------------
    # Sim-node → global-node mapping (per-page chain hash)
    # ------------------------------------------------------------------

    def _chain_hash(self, sim_node: RadixNode) -> int:
        """Per-page chain hash of the prefix from root to *sim_node*'s end.

        Cached on the node and invalidated when the node's ``pages`` tuple
        is replaced (which is what happens during a split).  The cached
        value is *invariant* under splits of ancestors because the chain
        is computed page-by-page — the compression boundary doesn't
        affect the hash.
        """
        if sim_node.parent is None:
            return 0
        cached_id = getattr(sim_node, "_oracle_hash_pages_id", None)
        if cached_id == id(sim_node.pages):
            return sim_node._oracle_chain_hash  # type: ignore[attr-defined]
        parent_h = self._chain_hash(sim_node.parent)
        h = parent_h
        for p in sim_node.pages:
            h = hash((h, p))
        sim_node._oracle_chain_hash = h  # type: ignore[attr-defined]
        sim_node._oracle_hash_pages_id = id(sim_node.pages)  # type: ignore[attr-defined]
        return h

    def _global_node_for(self, sim_node: RadixNode) -> Optional[RadixNode]:
        """Look up the global radix node corresponding to *sim_node*'s prefix."""
        h = self._chain_hash(sim_node)
        return self.path_hash_to_global.get(h)

    def _next_future_req(self, sim_node: RadixNode, after_idx: int) -> Optional[int]:
        """Smallest request index > *after_idx* that visits this prefix.

        Returns ``None`` when the node is never accessed again.
        """
        gn = self._global_node_for(sim_node)
        if gn is None:
            return None
        return self._next_future_req_global(gn, after_idx)

    def _next_future_req_global(
        self, gn: RadixNode, after_idx: int
    ) -> Optional[int]:
        """Same as ``_next_future_req`` but takes a global tree node directly."""
        future = self.global_future_reqs.get(id(gn))
        if not future:
            return None
        i = bisect.bisect_right(future, after_idx)
        if i >= len(future):
            return None
        return future[i]

    # ------------------------------------------------------------------
    # Saved-time helpers
    # ------------------------------------------------------------------

    def _compute_saved(self, total_tokens: int, hit_tokens: int) -> float:
        """``saved_compute`` for a single request, in seconds (HBM-only)."""
        if self.gpu_flops is None or self.gpu_flops <= 0 or hit_tokens <= 0:
            return 0.0
        h = min(hit_tokens, total_tokens)
        flop = self.model.prefill_flop(total_tokens) - self.model.incremental_prefill_flop(
            total_tokens, h
        )
        return flop / self.gpu_flops

    # ------------------------------------------------------------------
    # Per-request planning (oracle: split + checkpoint at every depth
    # that the *future* will reuse)
    # ------------------------------------------------------------------

    def plan_request(
        self,
        tree: RadixTree,
        matched_nodes: List[RadixNode],
        remaining_pages: List[PageKey],
    ) -> RequestPlan:
        """Admit every remaining page; mamba state at every depth on
        the request path that some *future* request will visit."""
        if not remaining_pages:
            return RequestPlan(remaining=[])

        statuses: List[PageStatus] = [PageStatus.KV_ONLY] * len(remaining_pages)
        fork_point = False

        if tree.mamba_state_token_equiv == 0:
            return RequestPlan(remaining=statuses)

        current_idx = tree.clock - 1
        if current_idx < 0 or current_idx >= self.num_requests:
            return RequestPlan(remaining=statuses)
        pgs = self.req_pages[current_idx]
        if not pgs:
            return RequestPlan(remaining=statuses)

        threshold = self._min_mamba_admit_depth
        matched_depth = matched_nodes[-1].depth_tokens if matched_nodes else 0

        gnode = self.global_tree.root
        gi = 0
        while gi < len(pgs):
            ch = gnode.children.get(pgs[gi])
            if ch is None:
                break
            ok = True
            for j in range(len(ch.pages)):
                if gi + j >= len(pgs) or pgs[gi + j] != ch.pages[j]:
                    ok = False
                    break
            if not ok:
                break
            depth = ch.depth_tokens
            future_visit = self._next_future_req_global(ch, current_idx)
            if future_visit is not None and (
                threshold == 0 or depth >= threshold
            ):
                if depth <= matched_depth:
                    # Only the deepest matched node can receive mamba
                    # via the plan API; honour it when the future-reused
                    # depth coincides with the matched boundary.
                    if matched_nodes and depth == matched_depth:
                        fork_point = True
                else:
                    # Translate depth → page index within remaining_pages
                    # (snap up to the first boundary ≥ depth).
                    offset = depth - matched_depth
                    cum = 0
                    for i, p in enumerate(remaining_pages):
                        cum += len(p)
                        if cum >= offset:
                            statuses[i] = PageStatus.KV_AND_MAMBA
                            break
            gi += len(ch.pages)
            gnode = ch

        return RequestPlan(
            remaining=statuses, mamba_on_matched_parent=fork_point
        )

    # ------------------------------------------------------------------
    # Eviction (Bélády: latest-next-use first)
    # ------------------------------------------------------------------

    def _collect_candidates(
        self, tree: RadixTree
    ) -> List[Tuple[RadixNode, EvictOp]]:
        """Enumerate eviction candidates: leaves + mamba-state internal nodes.

        Mirrors ``BranchStrategy._collect_candidates`` so the granularity
        matches what the simulator's ``_evict_until_fit`` expects.
        """
        out: List[Tuple[RadixNode, EvictOp]] = []
        q: deque[RadixNode] = deque(tree.root.children.values())
        while q:
            node = q.popleft()
            if not node.children:
                out.append((node, "leaf"))
            elif node.has_mamba_state:
                out.append((node, "mamba"))
            q.extend(node.children.values())
        return out

    def select_eviction(
        self, tree: RadixTree
    ) -> Optional[Tuple[RadixNode, EvictOp]]:
        candidates = self._collect_candidates(tree)
        if not candidates:
            return None

        # Sync the current request index from the tree clock.  Splits
        # have already happened during simulate_request, and the eviction
        # loop runs after the new node is inserted, so clock - 1 is the
        # current request being processed.
        current_idx = tree.clock - 1

        # Sort key per Bélády: largest "next use index" first
        # (i.e. evict the candidate whose next reuse is furthest in the
        # future or never).  ``None`` next use → infinity (evict first).
        # Tie-break: prefer evicting candidates that free more capacity.
        def sort_key(c: Tuple[RadixNode, EvictOp]) -> Tuple[int, int]:
            node, op = c
            nxt = self._next_future_req(node, current_idx)
            score = nxt if nxt is not None else 10**18
            if op == "leaf":
                free = node.num_tokens
            else:
                free = tree.mamba_state_token_equiv
            # Larger score → evict first; among ties, free more capacity.
            return (-score, -free)

        victim = min(candidates, key=sort_key)
        return victim
