"""Compact JSONL logger for radix tree mutations.

Writes one JSON object per line.  Event types::

    R  – request start   {e,rid,clk,ntok}
    SP – node split      {e,oid,pid,ppid,plen,sid,slen}
    H  – cache hit node  {e,id,ac,la[,crf]}
    I  – node inserted   {e,id,pid,len}
    MS – mamba state set {e,id}
    ME – mamba state evicted {e,id}
    E  – node(s) evicted {e,ids[,crf]}
    SS – stra snapshot   {e,nodes:[{id,<stra_key>:<val>,...},...]}
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
    model_name: str | None = None,
    dram_strategy: str | None = None,
    dram_capacity_spec: str | None = None,
    **_extra: object,
) -> str:
    """Build a deterministic filename encoding all simulation parameters.

    ``model_name`` is appended (when provided) so logs from different
    architectures don't collide.  ``dram_strategy`` / ``dram_capacity_spec``
    are appended when the run uses two-tier (HBM + DRAM) caching.
    """
    cap = str(capacity_spec).replace(" ", "")
    name = f"{dataset}_ps{page_size}_{ordering}_{strategy}_cap{cap}"
    if model_name:
        name += f"_{model_name}"
    if dram_strategy:
        dram_cap = str(dram_capacity_spec).replace(" ", "") if dram_capacity_spec else "0"
        name += f"_dram-{dram_strategy}_dcap{dram_cap}"
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
            "len": node.num_tokens,
        })

    def insert_raw(self, node_id: int, parent_id: int, num_tokens: int) -> None:
        """Write an insert event from captured ids (snapshot at insert time)."""
        self._write({"e": "I", "id": node_id, "pid": parent_id, "len": num_tokens})

    def split(self, old_id: int, prefix_id: int, prefix_pid: int,
              prefix_len: int, suffix_id: int, suffix_len: int) -> None:
        self._write({
            "e": "SP",
            "oid": old_id,
            "pid": prefix_id,
            "ppid": prefix_pid,
            "plen": prefix_len,
            "sid": suffix_id,
            "slen": suffix_len,
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
        # Per-node _stra_* snapshot so the viewer can show strategy state
        # on the ghost (just-evicted) node before it's pruned.
        stra_list = []
        any_stra = False
        for n in removed_nodes:
            s = {k[len("_stra_"):]: v for k, v in n.__dict__.items() if k.startswith("_stra_")}
            if s:
                any_stra = True
            stra_list.append(s)
        if any_stra:
            ev["stra"] = stra_list
        self._write(ev)

    def stra_snapshot(self, tree) -> None:
        """Dump every node's ``_stra_*`` attributes as one event.

        Called by the simulator after an eviction pass in which at least one
        node was removed.  Keys lose the ``_stra_`` prefix in the output.
        """
        entries = []
        for n in tree.iter_nodes():
            stra = {
                k[len("_stra_"):]: v
                for k, v in n.__dict__.items()
                if k.startswith("_stra_")
            }
            if stra:
                entries.append({"id": n.creation_order, **stra})
        if entries:
            self._write({"e": "SS", "nodes": entries})

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
