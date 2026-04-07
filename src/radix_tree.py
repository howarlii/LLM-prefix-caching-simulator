"""Page-granularity radix tree for prefix KV cache simulation.

Compressed multi-page nodes
---------------------------
Each node stores one or more consecutive pages (a compressed Patricia trie).
When a new request diverges mid-node, the node is split at the divergence
point.  This dramatically reduces chain length when ``page_size`` is small
(e.g. 1), since an entire suffix is stored in a single node rather than
one node per page.

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
from typing import Dict, Iterator, List, Optional, Set, Tuple

PageKey = Tuple[int, ...]


@dataclass
class RadixNode:
    """One node = one or more consecutive cached pages."""

    pages: Tuple[PageKey, ...]
    num_tokens: int = 0
    parent: Optional[RadixNode] = None
    children: Dict[PageKey, RadixNode] = field(default_factory=dict)
    last_access: int = 0
    access_count: int = 0
    creation_order: int = 0
    # True if some request ended at this node (multi-turn "previous turn" boundary).
    is_turn_end: bool = False
    # True if a Mamba SSM state is stored at this node (hybrid-model mode).
    has_mamba_state: bool = False

    # Identity-based hash so nodes can live in sets.
    __hash__ = object.__hash__

    @property
    def num_pages(self) -> int:
        return len(self.pages)

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
        self._root = RadixNode(pages=())
        self._node_counter = 0
        self._clock = 0  # monotonic logical timestamp (incremented per request)
        self._token_count = 0
        self._mamba_state_token_equiv = mamba_state_token_equiv
        self._mamba_state_count = 0  # nodes currently carrying a Mamba state
        # Incremental indexes — avoid full-tree BFS on every eviction call.
        self._leaf_set: Set[RadixNode] = set()
        self._mamba_state_nodes: Set[RadixNode] = set()
        # Split log for external consumers (e.g. tree logger / viewer).
        # Each entry: (old_id, prefix_id, prefix_pid, prefix_len, suffix_id, suffix_len)
        self._pending_splits: List[Tuple[int, int, int, int, int, int]] = []

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
            self._mamba_state_nodes.add(node)

    def evict_mamba_state(self, node: RadixNode) -> None:
        """Drop only the Mamba state of *node*, keeping its KV cache (idempotent)."""
        if node.has_mamba_state:
            node.has_mamba_state = False
            self._mamba_state_count -= 1
            self._mamba_state_nodes.discard(node)

    def _add_token_count(self, delta: int) -> None:
        self._token_count += delta

    def _remove_node_accounting(self, node: RadixNode) -> None:
        """Subtract KV tokens and Mamba state for a node being removed."""
        self._token_count -= node.num_tokens
        if node.has_mamba_state:
            self._mamba_state_count -= 1
            self._mamba_state_nodes.discard(node)

    # ------------------------------------------------------------------
    # Node splitting
    # ------------------------------------------------------------------

    def split_node(self, node: RadixNode, split_at: int) -> RadixNode:
        """Split *node* at page boundary *split_at*.

        Mutates *node* in place to become the prefix (first *split_at* pages).
        Creates and returns a new suffix child that inherits the original
        node's children, ``is_turn_end``, and ``has_mamba_state``.

        The parent's ``children`` dict key remains valid because the prefix
        keeps the same first page.

        Returns the suffix node.
        """
        assert 0 < split_at < node.num_pages

        old_id = node.creation_order  # capture before mutation
        prefix_pages = node.pages[:split_at]
        suffix_pages = node.pages[split_at:]
        prefix_tokens = sum(len(p) for p in prefix_pages)
        suffix_tokens = node.num_tokens - prefix_tokens

        suffix = RadixNode(
            pages=suffix_pages,
            num_tokens=suffix_tokens,
            parent=node,
            children=node.children,
            last_access=node.last_access,
            access_count=node.access_count,
            creation_order=node.creation_order,
            is_turn_end=node.is_turn_end,
            has_mamba_state=node.has_mamba_state,
        )

        # Update grandchildren's parent pointers.
        for child in suffix.children.values():
            child.parent = suffix

        # Mutate node to become the prefix.
        node.pages = prefix_pages
        node.num_tokens = prefix_tokens
        node.children = {suffix_pages[0]: suffix}
        node.creation_order = self._next_creation_order()
        node.is_turn_end = False
        node.has_mamba_state = False

        # Update leaf tracking: node had leaf status iff it was a leaf before.
        # After split, node has a child (suffix) so it is no longer a leaf.
        # Suffix is a leaf iff the original node was a leaf.
        if node in self._leaf_set:
            self._leaf_set.discard(node)
            self._leaf_set.add(suffix)

        # Update mamba tracking: mamba state moves to suffix.
        if suffix.has_mamba_state:
            self._mamba_state_nodes.discard(node)
            self._mamba_state_nodes.add(suffix)

        # Record for logger: (old_id, prefix_id, prefix_pid, prefix_len, suffix_id, suffix_len)
        parent_id = node.parent.creation_order if node.parent else 0
        self._pending_splits.append((
            old_id, node.creation_order, parent_id,
            node.num_tokens, suffix.creation_order, suffix.num_tokens,
        ))

        return suffix

    def drain_pending_splits(self) -> List[Tuple[int, int, int, int, int, int]]:
        """Return and clear pending split records for logging.

        Each entry: ``(old_id, prefix_id, prefix_pid, prefix_len, suffix_id, suffix_len)``.
        """
        out = self._pending_splits
        self._pending_splits = []
        return out

    # ------------------------------------------------------------------
    # Request simulation
    # ------------------------------------------------------------------

    def simulate_request(
        self, pages: List[PageKey]
    ) -> Tuple[int, int, int, int, List[RadixNode], List[RadixNode]]:
        """Match longest page prefix, touch hit nodes, insert missing suffix.

        Returns
        -------
        hit_pages : int
        hit_tokens : int
        total_input_tokens : int
        turn_hit_tokens : int
        new_nodes : List[RadixNode]
        matched_nodes : List[RadixNode]
        """
        self._clock += 1
        ts = self._clock

        node = self._root
        matched_nodes: List[RadixNode] = []
        page_idx = 0

        while page_idx < len(pages):
            ch = node.children.get(pages[page_idx])
            if ch is None:
                break

            # Compare ch.pages against incoming pages.
            ch_pages = ch.pages
            match_len = 0
            for i in range(len(ch_pages)):
                if page_idx + i >= len(pages) or pages[page_idx + i] != ch_pages[i]:
                    break
                match_len += 1
            else:
                match_len = len(ch_pages)

            if match_len == len(ch_pages):
                # Full match of this node.
                ch.touch(ts)
                matched_nodes.append(ch)
                node = ch
                page_idx += len(ch_pages)
            else:
                # Partial match — split at the divergence point.
                self.split_node(ch, match_len)
                # After split, ch IS the prefix (mutated in place).
                ch.touch(ts)
                matched_nodes.append(ch)
                node = ch
                page_idx += match_len
                break

        # Build cumulative page/token counts for matched nodes.
        cum_pages: List[int] = []
        cum_tokens: List[int] = []
        cp, ct = 0, 0
        for n in matched_nodes:
            cp += n.num_pages
            ct += n.num_tokens
            cum_pages.append(cp)
            cum_tokens.append(ct)

        total_matched_pages = cp
        total_matched_tokens = ct

        # Determine effective hit depth (gated by Mamba state in hybrid mode).
        if self._mamba_state_token_equiv > 0:
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
        else:
            hit_pages = total_matched_pages
            hit_tokens = total_matched_tokens

        # Turn hit: tokens up to the deepest is_turn_end node within effective hit range.
        effective_range = (effective_hit_idx + 1) if self._mamba_state_token_equiv > 0 else len(matched_nodes)
        turn_hit_tokens = 0
        for i in range(effective_range):
            if matched_nodes[i].is_turn_end:
                turn_hit_tokens = cum_tokens[i]

        # Insert remaining suffix as a single compressed node.
        new_nodes: List[RadixNode] = []
        remaining = pages[page_idx:]
        if remaining:
            self._leaf_set.discard(node)
            order = self._next_creation_order()
            child = RadixNode(
                pages=tuple(remaining),
                num_tokens=sum(len(p) for p in remaining),
                parent=node,
                creation_order=order,
            )
            child.touch(ts)
            node.children[remaining[0]] = child
            self._add_token_count(child.num_tokens)
            new_nodes.append(child)
            self._leaf_set.add(child)
            node = child

        node.is_turn_end = True

        total_tokens = sum(len(x) for x in pages)
        return hit_pages, hit_tokens, total_tokens, turn_hit_tokens, new_nodes, matched_nodes

    # ------------------------------------------------------------------
    # Leaf / mamba accessors (backed by incremental indexes)
    # ------------------------------------------------------------------

    def leaf_nodes(self) -> List[RadixNode]:
        """All leaves (eviction candidates). Root is never evicted."""
        return list(self._leaf_set)

    def leaf_node_set(self) -> Set[RadixNode]:
        """Direct access to the leaf set (read-only view — do not mutate)."""
        return self._leaf_set

    def nodes_with_mamba_state(self) -> List[RadixNode]:
        """All non-root nodes that currently carry a Mamba state."""
        return list(self._mamba_state_nodes)

    def mamba_state_node_set(self) -> Set[RadixNode]:
        """Direct access to the mamba state node set (read-only view — do not mutate)."""
        return self._mamba_state_nodes

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove_leaf(self, node: RadixNode) -> List[RadixNode]:
        """Remove a leaf and prune ancestors until a branch or root is reached.

        Returns the list of all removed nodes (the leaf plus any pruned
        ancestors), useful for logging / debugging.
        """
        if node is self._root or not node.is_leaf():
            return []
        self._leaf_set.discard(node)
        self._remove_node_accounting(node)
        parent = node.parent
        if parent is None or node.pages[0] not in parent.children:
            return [node]
        del parent.children[node.pages[0]]
        pruned, survivor = self._prune_empty_chain(parent)
        # The survivor may have become a leaf after pruning.
        if survivor is not self._root and survivor.is_leaf():
            self._leaf_set.add(survivor)
        return [node] + pruned

    def _prune_empty_chain(self, node: RadixNode) -> Tuple[List[RadixNode], RadixNode]:
        """Prune childless ancestors. Returns (removed_nodes, surviving_ancestor)."""
        removed: List[RadixNode] = []
        cur: RadixNode = node
        while cur is not self._root and len(cur.children) == 0:
            # In hybrid mode, preserve nodes that carry a Mamba state —
            # they serve as recomputation checkpoints for descendants.
            if self._mamba_state_token_equiv > 0 and cur.has_mamba_state:
                break
            self._remove_node_accounting(cur)
            parent = cur.parent
            if parent is None:
                break
            pk = cur.pages[0]
            if pk in parent.children:
                del parent.children[pk]
            removed.append(cur)
            cur = parent
        return removed, cur

    # ------------------------------------------------------------------
    # Traversal utilities
    # ------------------------------------------------------------------

    def iter_nodes(self) -> Iterator[RadixNode]:
        """Breadth-first over all nodes except the empty root."""
        q: deque[RadixNode] = deque([self._root])
        while q:
            n = q.popleft()
            if n is not self._root:
                yield n
            q.extend(n.children.values())

    def depth_histogram(self) -> Dict[int, int]:
        """Map depth (in pages, 1 = first page below root) -> number of nodes."""
        hist: Dict[int, int] = {}
        q: deque[tuple[RadixNode, int]] = deque([(self._root, 0)])
        while q:
            n, depth = q.popleft()
            if n is not self._root:
                hist[depth] = hist.get(depth, 0) + 1
                depth += n.num_pages
            q.extend((c, depth) for c in n.children.values())
        return hist

    def valid_cached_depth_histogram(self) -> Dict[int, int]:
        """Per tree depth (pages), weighted branching mass."""
        hist: Dict[int, int] = {}
        q: deque[tuple[RadixNode, int]] = deque([(self._root, 0)])
        while q:
            n, depth = q.popleft()
            if n is not self._root:
                x = len(n.children)
                if x > 1:
                    hist[depth] = hist.get(depth, 0) + (x - 1)
                depth += n.num_pages
            q.extend((c, depth) for c in n.children.values())
        return hist

    def visit_counts_by_depth(self) -> Dict[int, List[int]]:
        """For each depth (pages), list of ``access_count`` values."""
        by_depth: Dict[int, List[int]] = {}
        q: deque[tuple[RadixNode, int]] = deque([(self._root, 0)])
        while q:
            n, depth = q.popleft()
            if n is not self._root:
                by_depth.setdefault(depth, []).append(n.access_count)
                depth += n.num_pages
            q.extend((c, depth) for c in n.children.values())
        return by_depth
