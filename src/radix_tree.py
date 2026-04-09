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
When the tree is constructed with a hybrid ``ModelConfig`` (i.e. one whose
``mamba_state_token_equiv > 0``), the tree operates in hybrid mode (e.g.
Mamba + full-attention).  Each node may optionally store a Mamba SSM state
(``has_mamba_state``).  In hybrid mode a cache *hit* at depth D requires the
node at depth D to carry a Mamba state; pages matched beyond the last such
node are counted as *KV-only hits* (KV cache present but no compute savings).
For pure-transformer models (or when no model is provided) the tree behaves
identically to the original full-attention simulation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from src.model_config import ModelConfig

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
    # Total tokens from root down to (and including) this node.  Set at
    # insertion / split time.  The root has depth_tokens == 0.
    depth_tokens: int = 0

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
    model:
        Optional :class:`~src.model_config.ModelConfig`.  When supplied,
        ``mamba_state_token_equiv`` is derived from it; pure-transformer
        models (or ``None``) collapse to full-attention behaviour with
        no recurrent-state overhead and no hit gating.
    """

    def __init__(self, model: Optional["ModelConfig"] = None) -> None:
        self._root = RadixNode(pages=())
        self._node_counter = 0
        self._clock = 0  # monotonic logical timestamp (incremented per request)
        self._token_count = 0
        self._model = model
        self._mamba_state_token_equiv = (
            model.mamba_state_token_equiv if model is not None else 0
        )
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
    def model(self) -> Optional["ModelConfig"]:
        """Model configuration used by this tree (``None`` for pure-transformer / unspecified)."""
        return self._model

    @property
    def mamba_state_token_equiv(self) -> int:
        """Token-equivalent cost of one Mamba SSM state (``0`` outside hybrid mode)."""
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
            depth_tokens=node.depth_tokens,  # suffix keeps the original total depth
        )

        # Update grandchildren's parent pointers.
        for child in suffix.children.values():
            child.parent = suffix

        # Mutate node to become the prefix.
        node.pages = prefix_pages
        node.num_tokens = prefix_tokens
        node.depth_tokens = node.depth_tokens - suffix_tokens  # prefix is shallower
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

    def match_and_split(
        self, pages: List[PageKey]
    ) -> Tuple[List[RadixNode], int]:
        """Walk the longest page prefix, splitting on partial node matches.

        Advances the logical clock and touches every fully-matched node
        (updates ``last_access`` and ``access_count``).  Splits the divergence
        node when only a prefix of its pages matches, so the returned
        ``matched_nodes`` always end on an exact page boundary.

        Does NOT insert any missing suffix — that is the simulator's job
        (driven by strategy-supplied ops via :meth:`insert_leaf_at`).

        Returns
        -------
        matched_nodes : List[RadixNode]
        page_idx : int
            Number of pages consumed by the match.
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

            ch_pages = ch.pages
            match_len = 0
            for i in range(len(ch_pages)):
                if page_idx + i >= len(pages) or pages[page_idx + i] != ch_pages[i]:
                    break
                match_len += 1
            else:
                match_len = len(ch_pages)

            if match_len == len(ch_pages):
                ch.touch(ts)
                matched_nodes.append(ch)
                node = ch
                page_idx += len(ch_pages)
            else:
                # Partial match — split at the divergence point.
                self.split_node(ch, match_len)
                ch.touch(ts)
                matched_nodes.append(ch)
                page_idx += match_len
                break

        return matched_nodes, page_idx

    def insert_leaf_at(
        self,
        parent: RadixNode,
        pages: Tuple[PageKey, ...],
        *,
        timestamp: Optional[int] = None,
        access_count: int = 1,
    ) -> RadixNode:
        """Insert *pages* as a single compressed leaf child of *parent*.

        The leaf is marked ``is_turn_end=True``.  Caller is responsible
        for ensuring *parent* does not already have a child with the
        same first-page key (typically by walking via ``match_and_split``
        and only inserting the leftover suffix).
        """
        assert pages, "insert_leaf_at requires at least one page"
        assert pages[0] not in parent.children, (
            "insert_leaf_at: parent already has a child with this first page"
        )
        ts = timestamp if timestamp is not None else self._clock
        self._leaf_set.discard(parent)
        order = self._next_creation_order()
        child_tokens = sum(len(p) for p in pages)
        child = RadixNode(
            pages=pages,
            num_tokens=child_tokens,
            parent=parent,
            creation_order=order,
            depth_tokens=parent.depth_tokens + child_tokens,
            access_count=access_count,
            last_access=ts,
            is_turn_end=True,
        )
        parent.children[pages[0]] = child
        self._add_token_count(child_tokens)
        self._leaf_set.add(child)
        return child

    def set_mamba_at_depth(
        self, pages: List[PageKey], depth: int
    ) -> Optional[RadixNode]:
        """Walk *pages* from root and set mamba state at the page-boundary
        node ending at the smallest depth ``>= depth`` along this path.

        If the requested depth falls inside a multi-page compressed
        node, the node is split at the smallest internal page boundary
        ``>= depth``.  This matches the mid-chain checkpoint behaviour
        of the original Marconi2/3 strategies.

        Returns the targeted node, or ``None`` if the path is shorter
        than ``depth`` or ``depth <= 0``.
        """
        if depth <= 0:
            return None
        node = self._root
        page_idx = 0
        cum = 0
        while page_idx < len(pages):
            ch = node.children.get(pages[page_idx])
            if ch is None:
                return None
            ch_end = cum + ch.num_tokens
            if ch_end >= depth:
                # The target lands inside (or at the end of) ``ch``.
                # Find the smallest page boundary at or past ``depth``.
                offset = depth - cum
                cum_in_ch = 0
                pages_to_keep = 0
                for p in ch.pages:
                    cum_in_ch += len(p)
                    pages_to_keep += 1
                    if cum_in_ch >= offset:
                        break
                if pages_to_keep < ch.num_pages and pages_to_keep > 0:
                    self.split_node(ch, pages_to_keep)
                # ``ch`` is now the prefix node ending at the chosen boundary.
                self.set_mamba_state(ch)
                return ch
            cum = ch_end
            page_idx += ch.num_pages
            node = ch
        return None

    # ------------------------------------------------------------------
    # Legacy: match + insert in one shot (used only by oracle/global-tree
    # builders that need to populate a tree from a request stream without
    # going through the strategy/op pipeline).
    # ------------------------------------------------------------------

    def simulate_request(self, pages: List[PageKey]) -> List[RadixNode]:
        """Walk + split + insert remaining suffix as a single leaf.

        Convenience wrapper around :meth:`match_and_split` and
        :meth:`insert_leaf_at`.  Used by the oracle global-tree
        constructors which don't need the strategy/op pipeline.

        Returns the matched_nodes list (the inserted leaf, if any, is
        appended).
        """
        matched_nodes, page_idx = self.match_and_split(pages)
        remaining = pages[page_idx:]
        if remaining:
            parent = matched_nodes[-1] if matched_nodes else self._root
            leaf = self.insert_leaf_at(parent, tuple(remaining))
            matched_nodes.append(leaf)
        elif matched_nodes:
            matched_nodes[-1].is_turn_end = True
        return matched_nodes

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
