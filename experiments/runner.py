"""Shared helpers to run simulations and persist JSON/CSV results."""

from __future__ import annotations

import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from tqdm import tqdm

from src.cache_simulator import KVCacheSimulator
from src.config import (
    DEFAULT_GPU_FLOPS,
    DEFAULT_PCIE_BANDWIDTH,
    DEFAULT_TOKENIZER_NAME,
    PREPARED_CACHE_DIR,
    gb_to_token_capacity,
    ensure_hf_cache_dirs,
)
from src.datasets_loader import is_mooncake_trace_dataset, load_raw_requests
from src.metrics import RunMetrics, compute_run_metrics
from src.model_config import DEFAULT_MODEL, ModelConfig
from src.request_generator import (
    OrderingName,
    TokenizedRequest,
    load_or_tokenize,
    order_requests,
)
from src.strategies import BranchStrategy, CRFDecouplingStrategy, FIFOStrategy, LRUStrategy, MarconiStrategy, Marconi2Strategy, Marconi3Strategy, OracleGreedyStrategy, EvictionStrategy


def effective_page_size(dataset: str, page_size: int) -> int:
    """Mooncake traces ship one hash id per KV block; only ``page_size == 1`` matches that layout."""
    if is_mooncake_trace_dataset(dataset):
        return 1
    return page_size


def strategy_from_name(
    name: str,
    model: ModelConfig = DEFAULT_MODEL,
    *,
    gpu_flops: Optional[float] = None,
    pcie_bandwidth: Optional[float] = None,
    requests: Optional[List["TokenizedRequest"]] = None,
    page_size: Optional[int] = None,
) -> EvictionStrategy:
    n = name.lower()
    # Oracle strategies need the full request sequence at construction
    # time.  ``requests`` and ``page_size`` are required for these names;
    # they are ignored by every other strategy.
    if n == "oracle_greedy" or n.startswith("oracle_greedy"):
        if requests is None or page_size is None:
            raise ValueError(
                "oracle_greedy strategy requires `requests` and `page_size` kwargs"
            )
        return OracleGreedyStrategy(
            requests_token_ids=[r.token_ids for r in requests],
            page_size=page_size,
            model=model,
            gpu_flops=gpu_flops,
            pcie_bandwidth=pcie_bandwidth,
        )
    # Hardware-aware kwargs for the "new" strategies (branch + marconi3) that
    # need gpu_flops/pcie_bandwidth to compute the mamba-state admit-depth
    # threshold.  Pure-LRU/LFU/FIFO/older-marconi don't accept them.
    hw_kwargs: dict = {}
    if gpu_flops is not None:
        hw_kwargs["gpu_flops"] = gpu_flops
    if pcie_bandwidth is not None:
        hw_kwargs["pcie_bandwidth"] = pcie_bandwidth

    if n == "lru":
        return LRUStrategy()
    if n == "fifo":
        return FIFOStrategy()
    if n == "branch":
        return BranchStrategy(model=model, **hw_kwargs)
    if n == "branch_nt":
        return BranchStrategy(newtouch=True, model=model, **hw_kwargs)
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
            **hw_kwargs,
            **kwargs,
        )
    if n == "marconi3" or n.startswith("marconi3_"):
        m = re.search(r"_a([\d.]+)", n)
        kwargs = {"alpha": float(m.group(1))} if m else {}
        return Marconi3Strategy(model=model, **hw_kwargs, **kwargs)
    if n == "crf_decoupling" or n.startswith("crf_decoupling_"):
        # Optional lambda suffix: "crf_decoupling_0.01" → lambda_decay=0.01
        parts = n.split("_", 2)
        lam = float(parts[2]) if len(parts) == 3 else 0.001
        return CRFDecouplingStrategy(lambda_decay=lam)
    raise ValueError(f"Unknown strategy {name!r}")


def capacity_from_spec(spec: str, kv_bytes_per_token: int | None = None, *, model: ModelConfig | None = None) -> Optional[int]:
    """Convert a capacity spec string to token count.

    ``"inf"`` / ``"none"`` / ``"unlimited"`` → ``None`` (unlimited).
    ``"0"`` → ``0`` (used to signal "DRAM tier disabled").
    Otherwise interpreted as gigabytes.
    """
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
    *,
    dram_strategy: Optional[EvictionStrategy] = None,
    dram_capacity_tokens: Optional[int] = 0,
    logger: object = None,
    model: Optional[ModelConfig] = None,
    gpu_flops: float = DEFAULT_GPU_FLOPS,
    pcie_bandwidth: float = DEFAULT_PCIE_BANDWIDTH,
) -> RunMetrics:
    """Run a single-tier or two-tier simulation.

    DRAM tier is enabled iff ``dram_strategy`` is not None and
    ``dram_capacity_tokens != 0``.  ``dram_capacity_tokens=None`` means
    unlimited DRAM (no eviction); a positive int caps the tier; ``0``
    disables DRAM.

    ``gpu_flops`` (FLOP/sec) and ``pcie_bandwidth`` (bytes/sec) parameterise
    the wall-clock cost model used by ``compute_run_metrics``.
    """
    sim = KVCacheSimulator(
        page_size=page_size,
        strategy=strategy,
        capacity_tokens=capacity_tokens,
        dram_strategy=dram_strategy,
        dram_capacity_tokens=dram_capacity_tokens,
        model=model,
        logger=logger,
        gpu_flops=gpu_flops,
        pcie_bandwidth=pcie_bandwidth,
    )
    desc = "Simulating" + (" (HBM+DRAM)" if sim.dram_enabled else "")
    for req in tqdm(requests, desc=desc, leave=False, disable=not sys.stderr.isatty()):
        sim.process_token_ids(req.token_ids)
    return compute_run_metrics(
        sim.state,
        sim.tree,
        model=model,
        dram_tree=sim.dram_tree,
        gpu_flops=gpu_flops,
        pcie_bandwidth=pcie_bandwidth,
    )


RESULT_CSV_FIELDS: List[str] = [
    # ── Run identity ────────────────────────────────────────────────────
    "dataset",
    "page_size",
    "ordering",
    "strategy",
    "capacity_spec",
    "dram_strategy",
    "dram_capacity_spec",
    "tokenizer",
    "model_name",
    "gpu_flops",
    "pcie_bandwidth",
    "num_requests",
    "total_input_tokens",
    # ── Tier hit rates ──────────────────────────────────────────────────
    "hbm_token_hit_rate",
    "dram_token_hit_rate",
    # ── Token / capacity summary ────────────────────────────────────────
    "load_tokens",
    "compute_tokens",
    "load_compute_ratio",
    "peak_cached_tokens",
    "avg_cached_tokens",
    "dram_peak_cached_tokens",
    "dram_avg_cached_tokens",
    "avg_promoted_tokens_per_req",
    "avg_demoted_tokens_per_req",
    # ── Branch statistics ───────────────────────────────────────────────
    "req_branch_rate",
    "req_new_branch_rate",
    # ── FLOP counts and savings ─────────────────────────────────────────
    "total_flop_no_cache",
    "total_flop_with_cache",
    "flop_save_rate",
    # ── Wall-clock time (seconds), GPU + PCIe ───────────────────────────
    "gpu_compute_time_no_cache",
    "gpu_compute_time_with_cache",
    "pcie_total_transfer_bytes",
    "pcie_total_transfer_time",
    "total_saved_time",
    "saved_time_rate",
    # ── Per-request saved-time distribution ─────────────────────────────
    "per_request_saved_time_mean",
    "per_request_saved_time_p50",
    "per_request_saved_time_p90",
    "per_request_saved_time_p99",
]


def persist_result_row(
    out_csv: Path,
    out_json_dir: Path,
    row: Dict[str, Any],
) -> None:
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dram_strategy = row.get("dram_strategy") or ""
    dram_cap_spec = row.get("dram_capacity_spec") or ""

    slug_parts = [
        f"{row.get('dataset')}",
        f"ps{row.get('page_size')}",
        f"{row.get('ordering')}",
        f"{row.get('strategy')}",
        f"cap{row.get('capacity_spec')}",
    ]
    if dram_strategy:
        slug_parts.append(f"dram-{dram_strategy}-{dram_cap_spec}")
    slug = "_".join(slug_parts)
    jpath = out_json_dir / f"{slug}.json"
    jpath.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = row.get("metrics") or {}
    flat: Dict[str, Any] = {
        "dataset": row.get("dataset"),
        "page_size": row.get("page_size"),
        "ordering": row.get("ordering"),
        "strategy": row.get("strategy"),
        "capacity_spec": row.get("capacity_spec"),
        "dram_strategy": dram_strategy,
        "dram_capacity_spec": dram_cap_spec,
        "tokenizer": row.get("tokenizer"),
        "model_name": row.get("model_name", ""),
        "gpu_flops": row.get("gpu_flops", ""),
        "pcie_bandwidth": row.get("pcie_bandwidth", ""),
    }
    # Pull every metric named in RESULT_CSV_FIELDS straight from the dict.
    for field in RESULT_CSV_FIELDS:
        if field in flat:
            continue
        flat[field] = metrics.get(field)

    KEY_FIELDS = (
        "dataset",
        "page_size",
        "ordering",
        "num_requests",
        "strategy",
        "capacity_spec",
        "dram_strategy",
        "dram_capacity_spec",
        "model_name",
        "gpu_flops",
        "pcie_bandwidth",
        "gpu_flops",
    )

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


def _prepared_cache_path(
    dataset: str,
    ordering: str,
    tokenizer_name: str,
    *,
    seed: int,
    max_requests: Optional[int],
    sessions_per_second: float,
    words_per_min: float,
    narrativeqa_docs: int,
    sharegpt_conversations: int,
) -> Path:
    """Path of the post-tokenize, post-order pickle cache for one prep call."""
    safe_tok = tokenizer_name.replace("/", "_")
    n_label = "all" if max_requests is None else str(int(max_requests))
    name = (
        f"{dataset}__{ordering}__{safe_tok}"
        f"__seed{seed}__n{n_label}"
        f"__sps{sessions_per_second}__wpm{words_per_min}"
        f"__nqa{narrativeqa_docs}__sgc{sharegpt_conversations}"
        f".pkl"
    )
    return PREPARED_CACHE_DIR / name


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
    # ── Fast path: full prepared-list pickle cache ───────────────────
    # Skips HF dataset loading + tokenization + ordering when a previous
    # call with identical inputs has already been persisted.
    cache_path = _prepared_cache_path(
        dataset, ordering, tokenizer_name,
        seed=seed,
        max_requests=max_requests,
        sessions_per_second=sessions_per_second,
        words_per_min=words_per_min,
        narrativeqa_docs=narrativeqa_docs,
        sharegpt_conversations=sharegpt_conversations,
    )
    if not force_retokenize and cache_path.is_file():
        try:
            with cache_path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # corrupted cache → fall through and regenerate

    ensure_hf_cache_dirs()
    raw = load_raw_requests(
        dataset,
        narrativeqa_docs=narrativeqa_docs,
        sharegpt_conversations=sharegpt_conversations,
        seed=seed,
        max_requests=max_requests,
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
        result = order_requests(tok_moon, **order_kw)
    else:
        tok = load_or_tokenize(
            dataset,
            raw,
            tokenizer_name=tokenizer_name,
            num_workers=tokenize_workers,
            force_recompute=force_retokenize,
        )
        result = order_requests(tok, **order_kw)

    # Persist for next time (best-effort).
    try:
        PREPARED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
    except Exception:
        pass

    return result
