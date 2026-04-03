"""Shared helpers to run simulations and persist JSON/CSV results."""

from __future__ import annotations

import csv
import json
import sys
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
from src.strategies import FIFOStrategy, LFUStrategy, LRUStrategy, MarconiStrategy, EvictionStrategy


def effective_page_size(dataset: str, page_size: int) -> int:
    """Mooncake traces ship one hash id per KV block; only ``page_size == 1`` matches that layout."""
    if is_mooncake_trace_dataset(dataset):
        return 1
    return page_size


def strategy_from_name(name: str) -> EvictionStrategy:
    n = name.lower()
    if n == "lru":
        return LRUStrategy()
    if n == "lfu":
        return LFUStrategy()
    if n == "fifo":
        return FIFOStrategy()
    if n == "marconi" or n.startswith("marconi_"):
        # Optional alpha suffix: "marconi_2.0" → alpha=2.0
        parts = n.split("_", 1)
        alpha = float(parts[1]) if len(parts) == 2 else 1.0
        return MarconiStrategy(alpha=alpha)
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
    mamba_state_token_equiv: int = 0,
) -> RunMetrics:
    sim = KVCacheSimulator(
        page_size=page_size,
        strategy=strategy,
        capacity_tokens=capacity_tokens,
        mamba_state_token_equiv=mamba_state_token_equiv,
    )
    for req in tqdm(requests, desc="Simulating", leave=False, disable=not sys.stderr.isatty()):
        sim.process_token_ids(req.token_ids)
    return compute_run_metrics(sim.state, sim.tree)


RESULT_CSV_FIELDS: List[str] = [
    "dataset",
    "page_size",
    "ordering",
    "strategy",
    "capacity_spec",
    "tokenizer",
    "mamba_state_token_equiv",
    "num_requests",
    "page_level_hit_rate",
    "token_level_hit_rate",
    "turn_level_hit_rate",
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
    "compute_savings_rate",
    "kv_only_hit_pages",
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
        "mamba_state_token_equiv": row.get("mamba_state_token_equiv", 0),
        "num_requests": metrics.get("num_requests"),
        "page_level_hit_rate": metrics.get("page_level_hit_rate"),
        "token_level_hit_rate": metrics.get("token_level_hit_rate"),
        "turn_level_hit_rate": metrics.get("turn_level_hit_rate"),
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
        "compute_savings_rate": metrics.get("compute_savings_rate"),
        "kv_only_hit_pages": metrics.get("kv_only_hit_pages"),
    }

    KEY_FIELDS = ("dataset", "page_size", "ordering", "strategy", "capacity_spec")

    new_row = {k: flat.get(k, "") for k in RESULT_CSV_FIELDS}
    key = tuple(str(new_row.get(k, "")) for k in KEY_FIELDS)

    if out_csv.is_file():
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)

        replaced = False
        for i, r in enumerate(existing):
            if tuple(str(r.get(k, "")) for k in KEY_FIELDS) == key:
                existing[i] = new_row
                replaced = True
                break

        if not replaced:
            existing.append(new_row)

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in RESULT_CSV_FIELDS} for r in existing)
    else:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerow(new_row)


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
