"""Shared helpers to run simulations and persist JSON/CSV results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from tqdm import tqdm

from src.cache_simulator import KVCacheSimulator
from src.config import (
    DEFAULT_TOKENIZER_NAME,
    gb_to_token_capacity,
    ensure_hf_cache_dirs,
)
from src.datasets_loader import is_mooncake_trace_dataset, load_raw_requests
from src.metrics import RunMetrics, compute_run_metrics
from src.request_generator import (
    OrderingName,
    TokenizedRequest,
    load_or_tokenize,
    order_requests,
)


def effective_page_size(dataset: str, page_size: int) -> int:
    """Mooncake traces ship one hash id per KV block; only ``page_size == 1`` matches that layout."""
    if is_mooncake_trace_dataset(dataset):
        return 1
    return page_size
from src.strategies import FIFOStrategy, LFUStrategy, LRUStrategy, EvictionStrategy


def strategy_from_name(name: str) -> EvictionStrategy:
    n = name.lower()
    if n == "lru":
        return LRUStrategy()
    if n == "lfu":
        return LFUStrategy()
    if n == "fifo":
        return FIFOStrategy()
    raise ValueError(f"Unknown strategy {name!r}")


def capacity_from_spec(spec: str, kv_bytes_per_token: int) -> Optional[int]:
    s = spec.strip().lower()
    if s in ("inf", "none", "unlimited"):
        return None
    gb = float(s.replace("gb", "").strip())
    return gb_to_token_capacity(gb, kv_bytes_per_token)


def run_simulation(
    requests: List[TokenizedRequest],
    page_size: int,
    strategy: EvictionStrategy,
    capacity_tokens: Optional[int],
) -> RunMetrics:
    sim = KVCacheSimulator(
        page_size=page_size,
        strategy=strategy,
        capacity_tokens=capacity_tokens,
    )
    for req in tqdm(requests, desc="Simulating", leave=False):
        sim.process_token_ids(req.token_ids)
    return compute_run_metrics(sim.state, sim.tree)


RESULT_CSV_FIELDS: List[str] = [
    "dataset",
    "page_size",
    "ordering",
    "strategy",
    "capacity_spec",
    "tokenizer",
    "num_requests",
    "page_level_hit_rate",
    "token_level_hit_rate",
    "per_request_hit_rate_mean",
    "per_request_hit_rate_p50",
    "per_request_hit_rate_p90",
    "per_request_hit_rate_p99",
    "load_tokens",
    "compute_tokens",
    "load_compute_ratio",
    "peak_cached_tokens",
    "avg_cached_tokens",
    "total_input_tokens",
    "tree_depth_histogram_json",
    "tree_access_by_depth_json",
]


def persist_result_row(
    out_csv: Path,
    out_json_dir: Path,
    row: Dict[str, Any],
) -> None:
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    slug = (
        f"{row.get('dataset')}_ps{row.get('page_size')}_"
        f"{row.get('ordering')}_{row.get('strategy')}_cap{row.get('capacity_spec')}"
    )
    jpath = out_json_dir / f"{slug}.json"
    jpath.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = row.get("metrics") or {}
    flat: Dict[str, Any] = {
        "dataset": row.get("dataset"),
        "page_size": row.get("page_size"),
        "ordering": row.get("ordering"),
        "strategy": row.get("strategy"),
        "capacity_spec": row.get("capacity_spec"),
        "tokenizer": row.get("tokenizer"),
        "num_requests": metrics.get("num_requests"),
        "page_level_hit_rate": metrics.get("page_level_hit_rate"),
        "token_level_hit_rate": metrics.get("token_level_hit_rate"),
        "per_request_hit_rate_mean": metrics.get("per_request_hit_rate_mean"),
        "per_request_hit_rate_p50": metrics.get("per_request_hit_rate_p50"),
        "per_request_hit_rate_p90": metrics.get("per_request_hit_rate_p90"),
        "per_request_hit_rate_p99": metrics.get("per_request_hit_rate_p99"),
        "load_tokens": metrics.get("load_tokens"),
        "compute_tokens": metrics.get("compute_tokens"),
        "load_compute_ratio": metrics.get("load_compute_ratio"),
        "peak_cached_tokens": metrics.get("peak_cached_tokens"),
        "avg_cached_tokens": metrics.get("avg_cached_tokens"),
        "total_input_tokens": metrics.get("total_input_tokens"),
        "tree_depth_histogram_json": json.dumps(
            metrics.get("tree_depth_histogram", {}), ensure_ascii=False
        ),
        "tree_access_by_depth_json": json.dumps(
            metrics.get("tree_access_by_depth", {}), ensure_ascii=False
        ),
    }

    write_header = not out_csv.is_file()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: flat.get(k, "") for k in RESULT_CSV_FIELDS})


def prepare_requests(
    dataset: str,
    ordering: str,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    *,
    narrativeqa_docs: int = 50,
    sharegpt_conversations: int = 10_000,
    seed: int = 0,
    tokenize_workers: int = 0,
    force_retokenize: bool = False,
    max_requests: Optional[int] = None,
) -> List[TokenizedRequest]:
    ensure_hf_cache_dirs()
    raw = load_raw_requests(
        dataset,
        narrativeqa_docs=narrativeqa_docs,
        sharegpt_conversations=sharegpt_conversations,
        seed=seed,
    )
    if not raw:
        return []
    if max_requests is not None:
        raw = raw[:max_requests]
    if is_mooncake_trace_dataset(dataset):
        tok_moon: List[TokenizedRequest] = []
        for r in raw:
            h = r.meta.get("hash_ids")
            if not isinstance(h, list) or not h:
                continue
            meta = dict(r.meta)
            meta.pop("hash_ids", None)
            tok_moon.append(
                TokenizedRequest(
                    token_ids=[int(x) for x in h],
                    group_id=r.group_id,
                    meta=meta,
                )
            )
        return order_requests(tok_moon, mode=cast(OrderingName, ordering), seed=seed)
    tok = load_or_tokenize(
        dataset,
        raw,
        tokenizer_name=tokenizer_name,
        num_workers=tokenize_workers,
        force_recompute=force_retokenize,
    )
    return order_requests(tok, mode=cast(OrderingName, ordering), seed=seed)
