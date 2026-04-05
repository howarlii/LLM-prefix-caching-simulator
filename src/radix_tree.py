"""Page-granularity radix tree for prefix KV cache simulation.

Hybrid-model extension
----------------------
When ``mamba_state_token_equiv > 0`` the tree operates in hybrid mode (e.g.
Mamba + full-attention).  Each node may optionally store a Mamba SSM state
(``has_mamba_state``).  In hybrid mode a cache *hit* at depth D requires the
node at depth D to carry a Mamba state; pages matched beyond the last such
node are counted as *KV-only hits* (KV cache present but no compute savings).
When ``mamba_state_token_equiv == 0`` (default) the tree behaves identically
to the original pure full-attention simulation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

PageKey = Tuple[int, ...]


@dataclass
class RadixNode:
    """One node = one cached page (variable length for the final partial page)."""

    page: PageKey
    parent: Optional[RadixNode] = None
    children: Dict[PageKey, RadixNode] = field(default_factory=dict)
    last_access: int = 0
    access_count: int = 0
    creation_order: int = 0
    # True if some request ended at this node (multi-turn "previous turn" boundary).
    is_turn_end: bool = False
    # True if a Mamba SSM state is stored at this node (hybrid-model mode).
    has_mamba_state: bool = False

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def touch(self, timestamp: int = 0) -> None:
        self.last_access = timestamp
        self.access_count += 1


class RadixTree:
    """Radix tree keyed by consecutive pages (tuples of token ids).

    Parameters
    ----------
    mamba_state_token_equiv:
        How many full-attention token-equivalent units one Mamba SSM state
        counts toward cache capacity.  ``0`` (default) = pure full-attention
        mode, no Mamba state overhead and no hit gating.
    """

    def __init__(self, mamba_state_token_equiv: int = 0) -> None:
        self._root = RadixNode(page=())
        self._node_counter = 0
        self._clock = 0  # monotonic logical timestamp (incremented per request)
        self._token_count = 0
        self._mamba_state_token_equiv = mamba_state_token_equiv
        self._mamba_state_count = 0  # nodes currently carrying a Mamba state

    @property
    def root(self) -> RadixNode:
        return self._root

    @property
    def clock(self) -> int:
        """Current logical timestamp (incremented once per ``simulate_request``)."""
        return self._clock

    @property
    def mamba_state_token_equiv(self) -> int:
        return self._mamba_state_token_equiv

    def _next_creation_order(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def total_cached_tokens(self) -> int:
        """Effective capacity used: KV tokens + Mamba state overhead."""
        return self._token_count + self._mamba_state_count * self._mamba_state_token_equiv

    def total_kv_tokens(self) -> int:
        """Raw KV token count (excludes Mamba state overhead)."""
        return self._token_count

    def total_mamba_states(self) -> int:
        """Number of nodes currently carrying a Mamba state."""
        return self._mamba_state_count

    def set_mamba_state(self, node: RadixNode) -> None:
        """Mark *node* as carrying a Mamba state (idempotent)."""
        if not node.has_mamba_state:
            node.has_mamba_state = True
            self._mamba_state_count += 1

    def evict_mamba_state(self, node: RadixNode) -> None:
        """Drop only the Mamba state of *node*, keeping its KV cache (idempotent)."""
        if node.has_mamba_state:
            node.has_mamba_state = False
            self._mamba_state_count -= 1

    def _add_token_count(self, delta: int) -> None:
        self._token_count += delta

    def _remove_node_accounting(self, node: RadixNode) -> None:
        """Subtract KV tokens and Mamba state for a node being removed."""
        self._token_count -= len(node.page)
        if node.has_mamba_state:
            self._mamba_state_count -= 1

    def _clear_turn_end_below(self, node: RadixNode) -> None:
        """No-op: once a node is marked as a turn end, it stays that way."""
        pass

    def simulate_request(
        self, pages: List[PageKey]
    ) -> Tuple[int, int, int, int, int, List[RadixNode], List[RadixNode]]:
        """Match longest page prefix, touch hit nodes, insert missing suffix.

        Returns
        -------
        hit_pages : int
            Pages that are fully usable from cache.
            *Pure full-attention*: all matched pages.
            *Hybrid*: pages up to (and including) the deepest matched node that
            carries a Mamba state.
        hit_tokens : int
            Token count corresponding to ``hit_pages``.
        kv_only_hit_pages : int
            Pages that are present in the KV cache but *beyond* the last Mamba
            state checkpoint — their KV is cached but computation cannot be
            skipped without the Mamba state.  Always 0 in pure full-attention
            mode.
        total_input_tokens : int
            Total tokens in this request.
        turn_hit_tokens : int
            Tokens for hit pages whose node was marked as the end of a prior
            request (conversation-turn continuation), gated by the same Mamba
            state rule as ``hit_pages``.
        new_nodes : List[RadixNode]
            Newly inserted nodes (suffix that was not cached).  The caller
            (simulator) uses these to apply the admission policy.
        matched_nodes : List[RadixNode]
            Nodes that were matched during the prefix walk (cache hits).
            Used by strategies that need to update per-node metadata on hits.
        """
        self._clock += 1
        ts = self._clock

        node = self._root
        matched_nodes: List[RadixNode] = []

        for p in pages:
            ch = node.children.get(p)
            if ch is None:
                break
            ch.touch(ts)
            matched_nodes.append(ch)
            node = ch

        match_depth = len(matched_nodes)

        # Determine effective hit depth (gated by Mamba state in hybrid mode).
        if self._mamba_state_token_equiv > 0:
            effective_hit = 0
            for i, n in enumerate(matched_nodes):
                if n.has_mamba_state:
                    effective_hit = i + 1
            hit_pages = effective_hit
            hit_tokens = sum(len(matched_nodes[i].page) for i in range(effective_hit))
            kv_only_hit_pages = match_depth - effective_hit
        else:
            effective_hit = match_depth
            hit_pages = match_depth
            hit_tokens = sum(len(n.page) for n in matched_nodes)
            kv_only_hit_pages = 0

        # Turn hit: tokens up to the deepest is_turn_end node within effective hit range.
        # A cache hit is only usable from a turn boundary; if the match ends mid-turn,
        # walk back to the nearest turn_end ancestor (analogous to Mamba state gating).
        effective_turn_end = 0
        for i in range(effective_hit):
            if matched_nodes[i].is_turn_end:
                effective_turn_end = i + 1
        turn_hit_tokens = sum(len(matched_nodes[i].page) for i in range(effective_turn_end))

        # Insert suffix pages (always starting from full match depth, not effective hit).
        new_nodes: List[RadixNode] = []
        for p in pages[match_depth:]:
            order = self._next_creation_order()
            child = RadixNode(
                page=p,
                parent=node,
                creation_order=order,
            )
            child.touch(ts)
            node.children[p] = child
            self._add_token_count(len(p))
            new_nodes.append(child)
            node = child

        self._clear_turn_end_below(node)
        node.is_turn_end = True

        total_tokens = sum(len(x) for x in pages)
        return hit_pages, hit_tokens, kv_only_hit_pages, total_tokens, turn_hit_tokens, new_nodes, matched_nodes

    def leaf_nodes(self) -> List[RadixNode]:
        """All leaves (eviction candidates). Root is never evicted."""
        out: List[RadixNode] = []
        q: deque[RadixNode] = deque(self._root.children.values())
        while q:
            n = q.popleft()
            if n.is_leaf():
                out.append(n)
            else:
                q.extend(n.children.values())
        return out

    def nodes_with_mamba_state(self) -> List[RadixNode]:
        """All non-root nodes that currently carry a Mamba state."""
        out: List[RadixNode] = []
        q: deque[RadixNode] = deque(self._root.children.values())
        while q:
            n = q.popleft()
            if n.has_mamba_state:
                out.append(n)
            q.extend(n.children.values())
        return out

    def remove_leaf(self, node: RadixNode) -> List[RadixNode]:
        """Remove a leaf and prune ancestors until a branch or root is reached.

        Returns the list of all removed nodes (the leaf plus any pruned
        ancestors), useful for logging / debugging.
        """
        if node is self._root or not node.is_leaf():
            return []
        self._remove_node_accounting(node)
        parent = node.parent
        if parent is None or node.page not in parent.children:
            return [node]
        del parent.children[node.page]
        pruned = self._prune_empty_chain(parent)
        return [node] + pruned

    def _prune_empty_chain(self, node: RadixNode) -> List[RadixNode]:
        removed: List[RadixNode] = []
        cur: Optional[RadixNode] = node
        while cur is not None and cur is not self._root and len(cur.children) == 0:
            # In hybrid mode, preserve nodes that carry a Mamba state —
            # they serve as recomputation checkpoints for descendants.
            if self._mamba_state_token_equiv > 0 and cur.has_mamba_state:
                break
            self._remove_node_accounting(cur)
            parent = cur.parent
            if parent is None:
                break
            pk = cur.page
            if pk in parent.children:
                del parent.children[pk]
            removed.append(cur)
            cur = parent
        return removed

    def iter_nodes(self) -> Iterator[RadixNode]:
        """Breadth-first over all nodes except the empty root (no stack overflow on deep chains)."""
        q: deque[RadixNode] = deque([self._root])
        while q:
            n = q.popleft()
            if n is not self._root:
                yield n
            q.extend(n.children.values())

    def depth_histogram(self) -> Dict[int, int]:
        """Map depth (1 = first page below root) -> number of nodes."""

        hist: Dict[int, int] = {}
        q: deque[tuple[RadixNode, int]] = deque([(self._root, 0)])
        while q:
            n, depth = q.popleft()
            if n is not self._root:
                hist[depth] = hist.get(depth, 0) + 1
                depth += 1
            q.extend((c, depth) for c in n.children.values())
        return hist

    def valid_cached_depth_histogram(self) -> Dict[int, int]:
        """Per tree depth, weighted branching mass: nodes with x>1 children add (x-1) at that depth."""

        hist: Dict[int, int] = {}
        q: deque[tuple[RadixNode, int]] = deque([(self._root, 0)])
        while q:
            n, depth = q.popleft()
            if n is not self._root:
                x = len(n.children)
                if x > 1:
                    hist[depth] = hist.get(depth, 0) + (x - 1)
                depth += 1
            q.extend((c, depth) for c in n.children.values())
        return hist

    def visit_counts_by_depth(self) -> Dict[int, List[int]]:
        """For each depth, list of ``access_count`` values (for analysis)."""
        by_depth: Dict[int, List[int]] = {}
        q: deque[tuple[RadixNode, int]] = deque([(self._root, 0)])
        while q:
            n, depth = q.popleft()
            if n is not self._root:
                by_depth.setdefault(depth, []).append(n.access_count)
                depth += 1
            q.extend((c, depth) for c in n.children.values())
        return by_depth
