"""Shared helpers to run simulations and persist JSON/CSV results."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from tqdm import tqdm

from src.cache_simulator import KVCacheSimulator, MultiTierCacheSimulator
from src.config import (
    DEFAULT_TOKENIZER_NAME,
    gb_to_token_capacity,
    ensure_hf_cache_dirs,
)
from src.datasets_loader import is_mooncake_trace_dataset, load_raw_requests
from src.metrics import (
    MultiTierRunMetrics,
    RunMetrics,
    compute_multi_tier_run_metrics,
    compute_run_metrics,
)
from src.model_config import DEFAULT_MODEL, ModelConfig
from src.request_generator import (
    OrderingName,
    TokenizedRequest,
    load_or_tokenize,
    order_requests,
)
from src.strategies import BranchStrategy, CRFDecouplingStrategy, FIFOStrategy, LFUStrategy, LRUStrategy, MarconiStrategy, Marconi2Strategy, Marconi3Strategy, EvictionStrategy


def effective_page_size(dataset: str, page_size: int) -> int:
    """Mooncake traces ship one hash id per KV block; only ``page_size == 1`` matches that layout."""
    if is_mooncake_trace_dataset(dataset):
        return 1
    return page_size


def strategy_from_name(name: str, model: ModelConfig = DEFAULT_MODEL) -> EvictionStrategy:
    n = name.lower()
    if n == "lru":
        return LRUStrategy()
    if n == "lfu":
        return LFUStrategy()
    if n == "fifo":
        return FIFOStrategy()
    if n == "branch":
        return BranchStrategy()
    if n == "branch_nt":
        return BranchStrategy(newtouch=True)
    if n == "marconi" or n.startswith("marconi_"):
        # Optional alpha suffix: "marconi_a2.0" → alpha=2.0
        m = re.search(r"_a([\d.]+)", n)
        kwargs: dict = {"alpha": float(m.group(1))} if m else {}
        return MarconiStrategy(model=model, **kwargs)
    # Marconi2 ablation variants: marconi2_e{0|1}_mn{0|1}[_a<float>]
    # e0 = root-relative evict (original), e1 = checkpoint-relative evict (new)
    # mn0 = no mid-chain mamba, mn1 = mid-chain mamba (new)
    m2_ablation = re.match(r"^marconi2_e([01])_mn([01])", n)
    if m2_ablation:
        use_evict = m2_ablation.group(1) == "1"
        use_mn = m2_ablation.group(2) == "1"
        m_alpha = re.search(r"_a([\d.]+)", n[m2_ablation.end():])
        kwargs = {"alpha": float(m_alpha.group(1))} if m_alpha else {}
        return Marconi2Strategy(use_checkpoint_relative_evict=use_evict, use_mid_chain_checkpoint=use_mn, **kwargs)
    if n == "marconi2" or n.startswith("marconi2_"):
        m = re.search(r"_a([\d.]+)", n)
        kwargs = {"alpha": float(m.group(1))} if m else {}
        return Marconi2Strategy(**kwargs)
    # Marconi3 ablation variants: marconi3_ev{N}[_mn{0|1}][_nt][_a<float>]
    # _mn defaults to 0 (mid-chain checkpoint disabled) when omitted.
    # _nt (optional) enables the selective newtouch LRU policy.
    m3_ablation = re.match(r"^marconi3_ev(\d)", n)
    if m3_ablation:
        evict_mode = f"ev{m3_ablation.group(1)}"
        rest = n[m3_ablation.end():]
        m_mn = re.search(r"_mn([01])", rest)
        use_mn = bool(m_mn) and m_mn.group(1) == "1"
        newtouch = "_nt" in rest
        m_alpha = re.search(r"_a([\d.]+)", rest)
        kwargs = {"alpha": float(m_alpha.group(1))} if m_alpha else {}
        return Marconi3Strategy(
            evict_mode=evict_mode,
            use_mid_chain_checkpoint=use_mn,
            newtouch=newtouch,
            model=model,
            **kwargs,
        )
    if n == "marconi3" or n.startswith("marconi3_"):
        m = re.search(r"_a([\d.]+)", n)
        kwargs = {"alpha": float(m.group(1))} if m else {}
        return Marconi3Strategy(model=model, **kwargs)
    if n == "crf_decoupling" or n.startswith("crf_decoupling_"):
        # Optional lambda suffix: "crf_decoupling_0.01" → lambda_decay=0.01
        parts = n.split("_", 2)
        lam = float(parts[2]) if len(parts) == 3 else 0.001
        return CRFDecouplingStrategy(lambda_decay=lam)
    raise ValueError(f"Unknown strategy {name!r}")


def capacity_from_spec(spec: str, kv_bytes_per_token: int | None = None, *, model: ModelConfig | None = None) -> Optional[int]:
    s = spec.strip().lower()
    if s in ("inf", "none", "unlimited"):
        return None
    gb = float(s.replace("gb", "").strip())
    if model is not None:
        return model.gb_to_token_capacity(gb)
    if kv_bytes_per_token is None:
        raise ValueError("Either model or kv_bytes_per_token must be provided")
    return gb_to_token_capacity(gb, kv_bytes_per_token)


def run_simulation(
    requests: List[TokenizedRequest],
    page_size: int,
    strategy: EvictionStrategy,
    capacity_tokens: Optional[int],
    logger: object = None,
    model: Optional[ModelConfig] = None,
) -> RunMetrics:
    mamba = model.mamba_state_token_equiv if model is not None else 0
    sim = KVCacheSimulator(
        page_size=page_size,
        strategy=strategy,
        capacity_tokens=capacity_tokens,
        mamba_state_token_equiv=mamba,
        logger=logger,
    )
    for req in tqdm(requests, desc="Simulating", leave=False, disable=not sys.stderr.isatty()):
        sim.process_token_ids(req.token_ids)
    return compute_run_metrics(sim.state, sim.tree, model=model)


def run_multi_tier_simulation(
    requests: List[TokenizedRequest],
    page_size: int,
    hbm_strategy: EvictionStrategy,
    dram_strategy: EvictionStrategy,
    hbm_capacity_tokens: int,
    dram_capacity_tokens: int,
    model: Optional[ModelConfig] = None,
) -> MultiTierRunMetrics:
    """Run a two-tier (HBM + DRAM) cache simulation."""
    mamba = model.mamba_state_token_equiv if model is not None else 0
    sim = MultiTierCacheSimulator(
        page_size=page_size,
        hbm_strategy=hbm_strategy,
        dram_strategy=dram_strategy,
        hbm_capacity_tokens=hbm_capacity_tokens,
        dram_capacity_tokens=dram_capacity_tokens,
        mamba_state_token_equiv=mamba,
    )
    for req in tqdm(requests, desc="Simulating (multi-tier)", leave=False, disable=not sys.stderr.isatty()):
        sim.process_token_ids(req.token_ids)
    return compute_multi_tier_run_metrics(
        sim.state, sim.hbm_tree, sim.dram_tree, model=model
    )


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
    "req_branch_rate",
    "req_new_branch_rate",
    "flops_save_rate",
    "total_flops_no_cache",
    "total_flops_with_cache",
    "model_name",
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
        "req_branch_rate": metrics.get("req_branch_rate"),
        "req_new_branch_rate": metrics.get("req_new_branch_rate"),
        "flops_save_rate": metrics.get("flops_save_rate"),
        "total_flops_no_cache": metrics.get("total_flops_no_cache"),
        "total_flops_with_cache": metrics.get("total_flops_with_cache"),
        "model_name": row.get("model_name", ""),
    }

    KEY_FIELDS = ("dataset", "page_size", "ordering", "strategy", "capacity_spec", "num_req")

    new_row = {k: flat.get(k, "") for k in RESULT_CSV_FIELDS}
    key = tuple(str(new_row.get(k, "")) for k in KEY_FIELDS)

    if out_csv.is_file():
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            existing = list(reader)

        # Merge headers: keep RESULT_CSV_FIELDS order, then append any
        # old columns not present in RESULT_CSV_FIELDS (preserves extras).
        merged_fields = list(RESULT_CSV_FIELDS)
        for f in old_fields:
            if f not in merged_fields:
                merged_fields.append(f)

        replaced = False
        for i, r in enumerate(existing):
            if tuple(str(r.get(k, "")) for k in KEY_FIELDS) == key:
                existing[i] = new_row
                replaced = True
                break

        if not replaced:
            existing.append(new_row)

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in merged_fields} for r in existing)
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
    sessions_per_second: float = 1.0,
    words_per_min: float = 90.0,
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

    order_kw = dict(
        mode=cast(OrderingName, ordering),
        seed=seed,
        sessions_per_second=sessions_per_second,
        words_per_min=words_per_min,
    )

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
        return order_requests(tok_moon, **order_kw)
    tok = load_or_tokenize(
        dataset,
        raw,
        tokenizer_name=tokenizer_name,
        num_workers=tokenize_workers,
        force_recompute=force_retokenize,
    )
    return order_requests(tok, **order_kw)
