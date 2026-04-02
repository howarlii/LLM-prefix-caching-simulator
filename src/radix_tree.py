"""Page-granularity radix tree for prefix KV cache simulation."""

from __future__ import annotations

import time
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
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    creation_order: int = 0
    # True if some request ended at this node (multi-turn "previous turn" boundary).
    is_turn_end: bool = False

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def touch(self) -> None:
        self.last_access = time.time()
        self.access_count += 1


class RadixTree:
    """Radix tree keyed by consecutive pages (tuples of token ids)."""

    def __init__(self) -> None:
        self._root = RadixNode(page=())
        self._node_counter = 0
        self._token_count = 0

    @property
    def root(self) -> RadixNode:
        return self._root

    def _next_creation_order(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def total_cached_tokens(self) -> int:
        """Sum of token lengths over all non-root nodes."""
        return self._token_count

    def _add_token_count(self, delta: int) -> None:
        self._token_count += delta

    def _clear_turn_end_below(self, node: RadixNode) -> None:
        """Drop turn-end marks on strict descendants (new turn end is shallower)."""
        q: deque[RadixNode] = deque(node.children.values())
        while q:
            n = q.popleft()
            n.is_turn_end = False
            q.extend(n.children.values())

    def simulate_request(self, pages: List[PageKey]) -> Tuple[int, int, int, int]:
        """Match longest page prefix, touch hit nodes, insert missing suffix.

        Returns ``(hit_pages, hit_tokens, total_input_tokens, turn_hit_tokens)``.
        ``turn_hit_tokens`` counts tokens for hit pages whose child node was marked
        as the end of a prior request (continuing the same conversation prefix).
        """
        node = self._root
        hit_pages = 0
        hit_tokens = 0
        turn_hit_tokens = 0
        for p in pages:
            ch = node.children.get(p)
            if ch is None:
                break
            if ch.is_turn_end:
                turn_hit_tokens += len(p)
            ch.touch()
            hit_pages += 1
            hit_tokens += len(p)
            node = ch

        first_new = True
        for p in pages[hit_pages:]:
            if first_new and node.is_turn_end:
                node.is_turn_end = False
            first_new = False
            order = self._next_creation_order()
            child = RadixNode(
                page=p,
                parent=node,
                creation_order=order,
            )
            child.touch()
            node.children[p] = child
            self._add_token_count(len(p))
            node = child

        self._clear_turn_end_below(node)
        node.is_turn_end = True

        total_tokens = sum(len(x) for x in pages)
        return hit_pages, hit_tokens, total_tokens, turn_hit_tokens

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

    def remove_leaf(self, node: RadixNode) -> None:
        """Remove a leaf and prune ancestors until a branch or root is reached."""
        if node is self._root or not node.is_leaf():
            return
        self._add_token_count(-len(node.page))
        parent = node.parent
        if parent is None or node.page not in parent.children:
            return
        del parent.children[node.page]
        self._prune_empty_chain(parent)

    def _prune_empty_chain(self, node: RadixNode) -> None:
        cur: Optional[RadixNode] = node
        while cur is not None and cur is not self._root and len(cur.children) == 0:
            self._add_token_count(-len(cur.page))
            parent = cur.parent
            if parent is None:
                break
            pk = cur.page
            if pk in parent.children:
                del parent.children[pk]
            cur = parent

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
