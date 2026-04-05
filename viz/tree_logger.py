"""Compact JSONL logger for radix tree mutations.

Writes one JSON object per line.  Event types::

    R  – request start   {e,rid,clk,ntok}
    H  – cache hit node  {e,id,ac,la[,crf]}
    I  – node inserted   {e,id,pid,len}
    MS – mamba state set {e,id}
    ME – mamba state evicted {e,id}
    E  – node(s) evicted {e,ids[,crf]}
    D  – request done    {e,rid,hit,miss,cap}

Enable by passing a ``TreeLogger`` instance as the *logger* argument to
:class:`KVCacheSimulator`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, List, Union

from src.radix_tree import RadixNode


def _log_filename(
    dataset: str,
    strategy: str,
    page_size: int,
    capacity_spec: str,
    ordering: str = "original",
    mamba_equiv: int = 0,
    **_extra: object,
) -> str:
    """Build a deterministic filename encoding all simulation parameters."""
    cap = str(capacity_spec).replace(" ", "")
    name = f"{dataset}_ps{page_size}_{ordering}_{strategy}_cap{cap}"
    if mamba_equiv > 0:
        name += f"_mamba{mamba_equiv}"
    return name + ".jsonl"


class TreeLogger:
    """Write tree mutation events to a JSONL file.

    Parameters
    ----------
    path:
        Output file path (will be created / overwritten).
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._f: IO[str] = open(path, "w", buffering=1 << 16)  # 64 KiB buffer

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _node_crf(node: RadixNode) -> dict:
        crf = getattr(node, "_crf_value", None)
        if crf is not None:
            return {"crf": round(crf, 6)}
        return {}

    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    # -- event writers ----------------------------------------------------

    def request_start(self, rid: int, clock: int, total_tokens: int) -> None:
        self._write({"e": "R", "rid": rid, "clk": clock, "ntok": total_tokens})

    def hit(self, node: RadixNode) -> None:
        d: dict = {
            "e": "H",
            "id": node.creation_order,
            "ac": node.access_count,
            "la": node.last_access,
        }
        d.update(self._node_crf(node))
        self._write(d)

    def insert(self, node: RadixNode) -> None:
        pid = node.parent.creation_order if node.parent else 0
        self._write({
            "e": "I",
            "id": node.creation_order,
            "pid": pid,
            "len": len(node.page),
        })

    def mamba_set(self, node: RadixNode) -> None:
        d = {"e": "MS", "id": node.creation_order}
        d.update(self._node_crf(node))
        self._write(d)

    def mamba_evict(self, node: RadixNode) -> None:
        self._write({"e": "ME", "id": node.creation_order})

    def evict(self, removed_nodes: List[RadixNode]) -> None:
        ev: dict = {"e": "E", "ids": [n.creation_order for n in removed_nodes]}
        if removed_nodes:
            ev.update(self._node_crf(removed_nodes[0]))
        self._write(ev)

    def request_end(
        self, rid: int, hit_tokens: int, miss_tokens: int, cached_tokens: int
    ) -> None:
        self._write({
            "e": "D",
            "rid": rid,
            "hit": hit_tokens,
            "miss": miss_tokens,
            "cap": cached_tokens,
        })

    # -- lifecycle --------------------------------------------------------

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "TreeLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
