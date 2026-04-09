"""Aggregate metrics from simulation traces and radix tree.

Naming convention: ``flop`` = a *count* of floating-point operations
(extensive); ``flops`` = operations *per second* (rate, only used for
hardware throughput such as the ``gpu_flops`` argument).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.cache_simulator import SimulationState
from src.model_config import ModelConfig
from src.radix_tree import RadixTree


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@dataclass
class RunMetrics:
    """JSON-serializable summary for one experiment configuration.

    DRAM-related fields are zero/None when DRAM is disabled (single-tier).

    Time-based fields (``*_time``, ``saved_time*``) are in seconds and
    require both a ``model`` (for FLOP counts) and ``gpu_flops`` /
    ``pcie_bandwidth`` arguments to ``compute_run_metrics``.
    """

    # ── Tier-specific hit rates (token granularity) ─────────────────────────
    # In single-tier mode, dram_token_hit_rate == 0 and hbm_token_hit_rate
    # equals the total prefix-cache hit rate.
    hbm_token_hit_rate: float
    dram_token_hit_rate: float

    # ── Per-request saved-time distribution (seconds) ───────────────────────
    per_request_saved_time_mean: float
    per_request_saved_time_p50: float
    per_request_saved_time_p90: float
    per_request_saved_time_p99: float

    # ── Token-level summary ────────────────────────────────────────────────
    load_tokens: int            # tokens served from cache (HBM + DRAM)
    compute_tokens: int         # tokens that had to be re-computed
    load_compute_ratio: Optional[float]
    peak_cached_tokens: int     # HBM peak
    avg_cached_tokens: float    # HBM average
    num_requests: int
    total_input_tokens: int

    # ── HBM tree shape histograms ──────────────────────────────────────────
    tree_depth_histogram: Dict[str, int]
    valid_cached_depth_histogram: Dict[str, int]
    tree_access_by_depth: Dict[str, Any]
    access_percentiles_by_depth: Dict[str, Dict[str, float]]

    # ── Branch statistics ──────────────────────────────────────────────────
    req_branch_rate: float = 0.0
    req_new_branch_rate: float = 0.0

    # ── FLOP counts and savings ─────────────────────────────────────────────
    # ``total_flop_*`` are extensive counts (operations); ``flop_save_rate``
    # is the fraction of compute avoided by prefix caching.
    total_flop_no_cache: Optional[float] = None
    total_flop_with_cache: Optional[float] = None
    flop_save_rate: Optional[float] = None

    # ── Wall-clock time (seconds), GPU + PCIe ───────────────────────────────
    gpu_compute_time_no_cache: Optional[float] = None
    gpu_compute_time_with_cache: Optional[float] = None
    pcie_total_transfer_bytes: int = 0
    pcie_total_transfer_time: float = 0.0
    total_saved_time: Optional[float] = None
    saved_time_rate: Optional[float] = None

    # ── DRAM-tier capacity & flow (zero when DRAM disabled) ─────────────────
    dram_peak_cached_tokens: int = 0
    dram_avg_cached_tokens: float = 0.0
    avg_promoted_tokens_per_req: float = 0.0
    avg_promoted_nodes_per_req: float = 0.0
    avg_promoted_gb_per_req: Optional[float] = None
    total_promoted_tokens: int = 0
    total_promoted_nodes: int = 0
    avg_demoted_tokens_per_req: float = 0.0
    avg_demoted_nodes_per_req: float = 0.0
    avg_demoted_gb_per_req: Optional[float] = None
    total_demoted_tokens: int = 0
    total_demoted_nodes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _empty_metrics() -> RunMetrics:
    return RunMetrics(
        hbm_token_hit_rate=0.0,
        dram_token_hit_rate=0.0,
        per_request_saved_time_mean=0.0,
        per_request_saved_time_p50=0.0,
        per_request_saved_time_p90=0.0,
        per_request_saved_time_p99=0.0,
        load_tokens=0,
        compute_tokens=0,
        load_compute_ratio=None,
        peak_cached_tokens=0,
        avg_cached_tokens=0.0,
        num_requests=0,
        total_input_tokens=0,
        tree_depth_histogram={},
        valid_cached_depth_histogram={},
        tree_access_by_depth={},
        access_percentiles_by_depth={},
    )


def compute_run_metrics(
    state: SimulationState,
    tree: RadixTree,
    model: Optional[ModelConfig] = None,
    dram_tree: Optional[RadixTree] = None,
    *,
    gpu_flops: Optional[float] = None,
    pcie_bandwidth: Optional[float] = None,
) -> RunMetrics:
    """Aggregate per-request traces into a :class:`RunMetrics` summary.

    The optional ``dram_tree`` argument enables DRAM-tier metrics; when omitted
    (or None), only single-tier (HBM) metrics are reported.

    Wall-clock fields (``gpu_compute_time_*``, ``pcie_total_transfer_*``,
    ``total_saved_time``, ``per_request_saved_time_*``) are populated only
    when ``model`` AND ``gpu_flops`` are provided.  ``pcie_bandwidth`` is
    only consulted in two-tier mode (when DRAM hits exist).
    """
    traces = state.traces
    if not traces:
        return _empty_metrics()

    n_req = len(traces)
    total_in = sum(t.input_tokens for t in traces)
    load_tokens = sum(t.hit_tokens for t in traces)
    compute_tokens = sum(t.miss_tokens for t in traces)
    branch_count = sum(1 for t in traces if t.is_branch)
    new_branch_count = sum(1 for t in traces if t.is_new_branch)

    usage = state.usage_samples
    peak = max(usage) if usage else 0
    avg_u = sum(usage) / len(usage) if usage else 0.0

    if compute_tokens == 0:
        lcr_out: Optional[float] = None
    else:
        lcr_out = float(load_tokens / compute_tokens)

    dh = tree.depth_histogram()
    vch = tree.valid_cached_depth_histogram()
    vad = tree.visit_counts_by_depth()

    tree_depth_histogram = {str(k): v for k, v in sorted(dh.items())}
    valid_cached_depth_histogram = {str(k): v for k, v in sorted(vch.items())}
    tree_access_by_depth = {
        str(k): {"count": len(v), "sum_access": sum(v), "mean_access": (sum(v) / len(v) if v else 0.0)}
        for k, v in sorted(vad.items())
    }

    access_percentiles_by_depth: Dict[str, Dict[str, float]] = {}
    for k, v in sorted(vad.items()):
        if not v:
            continue
        sv = sorted(float(x) for x in v)
        access_percentiles_by_depth[str(k)] = {
            "min": sv[0],
            "p50": _percentile(sv, 50),
            "p90": _percentile(sv, 90),
            "p99": _percentile(sv, 99),
            "max": sv[-1],
        }

    # ── Tier-specific hit rates ─────────────────────────────────────────────
    hbm_hit_total = sum(t.hbm_hit_tokens for t in traces)
    dram_hit_total = sum(t.dram_hit_tokens for t in traces)
    hbm_hr = hbm_hit_total / total_in if total_in else 0.0
    dram_hr = dram_hit_total / total_in if total_in else 0.0

    # ── DRAM-tier capacity samples & flow ───────────────────────────────────
    dram_u = state.dram_usage_samples
    dram_peak = max(dram_u) if dram_u else 0
    dram_avg = sum(dram_u) / len(dram_u) if dram_u else 0.0

    total_promoted_tokens = sum(t.promoted_tokens for t in traces)
    total_promoted_nodes = sum(t.promoted_nodes for t in traces)
    total_demoted_tokens = sum(t.demoted_tokens for t in traces)
    total_demoted_nodes = sum(t.demoted_nodes for t in traces)

    avg_promoted_gb: Optional[float] = None
    avg_demoted_gb: Optional[float] = None
    if model is not None and model.kv_bytes_per_token > 0 and dram_tree is not None:
        bpt = model.kv_bytes_per_token
        avg_promoted_gb = (total_promoted_tokens / n_req) * bpt / (1024 ** 3)
        avg_demoted_gb = (total_demoted_tokens / n_req) * bpt / (1024 ** 3)

    # ── FLOP counts + wall-clock time savings ───────────────────────────────
    # All time-based metrics need a ModelConfig (for the FLOP formulas) AND
    # a positive ``gpu_flops``.  ``pcie_bandwidth`` only matters in two-tier
    # mode (when DRAM hits exist).
    #
    # The simulator's smart DRAM fallback (in KVCacheSimulator) already
    # excludes DRAM hits whose PCIe transfer would cost more than recomputing
    # the suffix.  As a result ``t.dram_hit_tokens`` only ever counts useful
    # DRAM hits, and the naive aggregate
    #     saved = no_cache - with_cache - transfer
    # is always >= 0 — no further per-request clamping or fallback is needed.
    total_flop_no_cache: Optional[float] = None
    total_flop_with_cache: Optional[float] = None
    flop_save_rate: Optional[float] = None
    gpu_compute_time_no_cache: Optional[float] = None
    gpu_compute_time_with_cache: Optional[float] = None
    pcie_total_transfer_bytes = 0
    pcie_total_transfer_time = 0.0
    total_saved_time: Optional[float] = None
    saved_time_rate: Optional[float] = None
    per_req_saved_time: List[float] = []

    if model is not None:
        bpt = model.kv_bytes_per_token
        can_time = gpu_flops is not None and gpu_flops > 0
        has_pcie = pcie_bandwidth is not None and pcie_bandwidth > 0

        no_cache_total = 0.0
        with_cache_total = 0.0
        for t in traces:
            flop_no_cache = model.prefill_flop(t.input_tokens)
            flop_with_cache = model.incremental_prefill_flop(
                t.input_tokens, t.hit_tokens
            )
            no_cache_total += flop_no_cache
            with_cache_total += flop_with_cache

            req_dram_bytes = t.dram_hit_tokens * bpt
            pcie_total_transfer_bytes += req_dram_bytes

            if can_time:
                req_compute_avoided = (flop_no_cache - flop_with_cache) / gpu_flops
                req_transfer = (
                    req_dram_bytes / pcie_bandwidth
                    if has_pcie and req_dram_bytes > 0
                    else 0.0
                )
                # Always >= 0 because the simulator already filtered DRAM
                # hits whose transfer cost would dominate.
                per_req_saved_time.append(req_compute_avoided - req_transfer)

        total_flop_no_cache = no_cache_total
        total_flop_with_cache = with_cache_total
        flop_save_rate = (
            1.0 - with_cache_total / no_cache_total if no_cache_total > 0 else 0.0
        )

        if can_time:
            gpu_compute_time_no_cache = no_cache_total / gpu_flops
            gpu_compute_time_with_cache = with_cache_total / gpu_flops
            pcie_total_transfer_time = (
                pcie_total_transfer_bytes / pcie_bandwidth if has_pcie else 0.0
            )
            total_saved_time = (
                gpu_compute_time_no_cache
                - gpu_compute_time_with_cache
                - pcie_total_transfer_time
            )
            saved_time_rate = (
                total_saved_time / gpu_compute_time_no_cache
                if gpu_compute_time_no_cache > 0
                else 0.0
            )

    pst_sorted = sorted(per_req_saved_time)
    if per_req_saved_time:
        pst_mean = sum(per_req_saved_time) / len(per_req_saved_time)
        pst_p50 = _percentile(pst_sorted, 50)
        pst_p90 = _percentile(pst_sorted, 90)
        pst_p99 = _percentile(pst_sorted, 99)
    else:
        pst_mean = pst_p50 = pst_p90 = pst_p99 = 0.0

    return RunMetrics(
        hbm_token_hit_rate=hbm_hr,
        dram_token_hit_rate=dram_hr,
        per_request_saved_time_mean=pst_mean,
        per_request_saved_time_p50=pst_p50,
        per_request_saved_time_p90=pst_p90,
        per_request_saved_time_p99=pst_p99,
        load_tokens=load_tokens,
        compute_tokens=compute_tokens,
        load_compute_ratio=lcr_out,
        peak_cached_tokens=peak,
        avg_cached_tokens=avg_u,
        num_requests=n_req,
        total_input_tokens=total_in,
        tree_depth_histogram=tree_depth_histogram,
        valid_cached_depth_histogram=valid_cached_depth_histogram,
        tree_access_by_depth=tree_access_by_depth,
        access_percentiles_by_depth=access_percentiles_by_depth,
        req_branch_rate=branch_count / n_req,
        req_new_branch_rate=new_branch_count / n_req,
        total_flop_no_cache=total_flop_no_cache,
        total_flop_with_cache=total_flop_with_cache,
        flop_save_rate=flop_save_rate,
        gpu_compute_time_no_cache=gpu_compute_time_no_cache,
        gpu_compute_time_with_cache=gpu_compute_time_with_cache,
        pcie_total_transfer_bytes=pcie_total_transfer_bytes,
        pcie_total_transfer_time=pcie_total_transfer_time,
        total_saved_time=total_saved_time,
        saved_time_rate=saved_time_rate,
        dram_peak_cached_tokens=dram_peak,
        dram_avg_cached_tokens=dram_avg,
        avg_promoted_tokens_per_req=total_promoted_tokens / n_req,
        avg_promoted_nodes_per_req=total_promoted_nodes / n_req,
        avg_promoted_gb_per_req=avg_promoted_gb,
        total_promoted_tokens=total_promoted_tokens,
        total_promoted_nodes=total_promoted_nodes,
        avg_demoted_tokens_per_req=total_demoted_tokens / n_req,
        avg_demoted_nodes_per_req=total_demoted_nodes / n_req,
        avg_demoted_gb_per_req=avg_demoted_gb,
        total_demoted_tokens=total_demoted_tokens,
        total_demoted_nodes=total_demoted_nodes,
    )
