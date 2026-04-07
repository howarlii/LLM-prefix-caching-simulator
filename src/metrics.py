"""Aggregate metrics from simulation traces and radix tree."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.cache_simulator import MultiTierSimulationState, SimulationState
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
    """JSON-serializable summary for one experiment configuration."""

    page_level_hit_rate: float
    token_level_hit_rate: float
    turn_level_hit_rate: float
    per_request_hit_rate_mean: float
    per_request_hit_rate_p50: float
    per_request_hit_rate_p90: float
    per_request_hit_rate_p99: float
    load_tokens: int
    compute_tokens: int
    load_compute_ratio: Optional[float]
    peak_cached_tokens: int
    avg_cached_tokens: float
    num_requests: int
    total_input_tokens: int
    tree_depth_histogram: Dict[str, int]
    valid_cached_depth_histogram: Dict[str, int]
    tree_access_by_depth: Dict[str, Any]
    # Per-depth min/p50/p90/p99/max of node access_count (for distribution plots).
    access_percentiles_by_depth: Dict[str, Dict[str, float]]
    # Fraction of total FLOPs saved by prefix caching compared to no cache.
    # Computed as 1 - with_cache_flops / no_cache_flops.
    # None when no ModelConfig is provided (legacy mode).
    # Fraction of requests that branch from an intermediate node (not extending a leaf).
    req_branch_rate: float = 0.0
    # Like req_branch_rate, but only counts branches where the branch-point has no Mamba state.
    req_new_branch_rate: float = 0.0
    flops_save_rate: Optional[float] = None
    # Absolute FLOPs values (for downstream analysis).
    total_flops_no_cache: Optional[float] = None
    total_flops_with_cache: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_run_metrics(
    state: SimulationState,
    tree: RadixTree,
    model: Optional[ModelConfig] = None,
) -> RunMetrics:
    traces = state.traces
    if not traces:
        return RunMetrics(
            page_level_hit_rate=0.0,
            token_level_hit_rate=0.0,
            turn_level_hit_rate=0.0,
            per_request_hit_rate_mean=0.0,
            per_request_hit_rate_p50=0.0,
            per_request_hit_rate_p90=0.0,
            per_request_hit_rate_p99=0.0,
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

    total_pages_needed = sum(t.total_pages for t in traces)
    hit_pages = sum(t.hit_pages for t in traces)
    total_in = sum(t.input_tokens for t in traces)
    load_tokens = sum(t.hit_tokens for t in traces)
    compute_tokens = sum(t.miss_tokens for t in traces)

    pr_rates = [t.per_request_token_hit_rate for t in traces]
    pr_sorted = sorted(pr_rates)

    usage = state.usage_samples
    peak = max(usage) if usage else 0
    avg_u = sum(usage) / len(usage) if usage else 0.0

    turn_hit_tokens = sum(t.turn_hit_tokens for t in traces)
    branch_count = sum(1 for t in traces if t.is_branch)
    new_branch_count = sum(1 for t in traces if t.is_new_branch)

    page_hr = hit_pages / total_pages_needed if total_pages_needed else 0.0
    tok_hr = load_tokens / total_in if total_in else 0.0
    turn_hr = turn_hit_tokens / total_in if total_in else 0.0
    lcr = load_tokens / compute_tokens if compute_tokens else float("inf")

    dh = tree.depth_histogram()
    vch = tree.valid_cached_depth_histogram()
    vad = tree.visit_counts_by_depth()

    lcr_out: Optional[float]
    if compute_tokens == 0:
        lcr_out = None
    elif math.isfinite(lcr):
        lcr_out = float(lcr)
    else:
        lcr_out = None

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

    # FLOPs save rate: requires a ModelConfig to compute.
    flops_save_rate: Optional[float] = None
    total_flops_no_cache: Optional[float] = None
    total_flops_with_cache: Optional[float] = None
    if model is not None and traces:
        no_cache = sum(model.prefill_flops(t.input_tokens) for t in traces)
        with_cache = sum(
            model.incremental_prefill_flops(t.input_tokens, t.hit_tokens)
            for t in traces
        )
        total_flops_no_cache = no_cache
        total_flops_with_cache = with_cache
        flops_save_rate = 1.0 - with_cache / no_cache if no_cache > 0 else 0.0

    return RunMetrics(
        page_level_hit_rate=page_hr,
        token_level_hit_rate=tok_hr,
        turn_level_hit_rate=turn_hr,
        per_request_hit_rate_mean=sum(pr_rates) / len(pr_rates),
        per_request_hit_rate_p50=_percentile(pr_sorted, 50),
        per_request_hit_rate_p90=_percentile(pr_sorted, 90),
        per_request_hit_rate_p99=_percentile(pr_sorted, 99),
        load_tokens=load_tokens,
        compute_tokens=compute_tokens,
        load_compute_ratio=lcr_out,
        peak_cached_tokens=peak,
        avg_cached_tokens=avg_u,
        num_requests=len(traces),
        total_input_tokens=total_in,
        tree_depth_histogram=tree_depth_histogram,
        valid_cached_depth_histogram=valid_cached_depth_histogram,
        tree_access_by_depth=tree_access_by_depth,
        access_percentiles_by_depth=access_percentiles_by_depth,
        req_branch_rate=branch_count / len(traces) if traces else 0.0,
        req_new_branch_rate=new_branch_count / len(traces) if traces else 0.0,
        flops_save_rate=flops_save_rate,
        total_flops_no_cache=total_flops_no_cache,
        total_flops_with_cache=total_flops_with_cache,
    )


# ---------------------------------------------------------------------------
# Multi-tier metrics
# ---------------------------------------------------------------------------


@dataclass
class MultiTierRunMetrics(RunMetrics):
    """Extended metrics for two-tier (HBM + DRAM) cache simulation."""

    # Tier-specific hit rates (fraction of total input tokens).
    hbm_token_hit_rate: float = 0.0
    dram_token_hit_rate: float = 0.0

    # Tier-specific capacity usage.
    hbm_peak_cached_tokens: int = 0
    hbm_avg_cached_tokens: float = 0.0
    dram_peak_cached_tokens: int = 0
    dram_avg_cached_tokens: float = 0.0

    # Cache flow: promotion (DRAM → HBM) per request.
    avg_promoted_tokens_per_req: float = 0.0
    avg_promoted_nodes_per_req: float = 0.0
    avg_promoted_gb_per_req: Optional[float] = None
    total_promoted_tokens: int = 0
    total_promoted_nodes: int = 0

    # Cache flow: demotion (HBM → DRAM) per request.
    avg_demoted_tokens_per_req: float = 0.0
    avg_demoted_nodes_per_req: float = 0.0
    avg_demoted_gb_per_req: Optional[float] = None
    total_demoted_tokens: int = 0
    total_demoted_nodes: int = 0


def compute_multi_tier_run_metrics(
    state: MultiTierSimulationState,
    hbm_tree: RadixTree,
    dram_tree: RadixTree,
    model: Optional[ModelConfig] = None,
) -> MultiTierRunMetrics:
    """Compute metrics for a two-tier simulation run."""
    traces = state.traces
    if not traces:
        return MultiTierRunMetrics(
            page_level_hit_rate=0.0,
            token_level_hit_rate=0.0,
            turn_level_hit_rate=0.0,
            per_request_hit_rate_mean=0.0,
            per_request_hit_rate_p50=0.0,
            per_request_hit_rate_p90=0.0,
            per_request_hit_rate_p99=0.0,
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

    n_req = len(traces)
    total_pages_needed = sum(t.total_pages for t in traces)
    hit_pages = sum(t.hit_pages for t in traces)
    total_in = sum(t.input_tokens for t in traces)
    load_tokens = sum(t.hit_tokens for t in traces)
    compute_tokens = sum(t.miss_tokens for t in traces)
    turn_hit_tokens = sum(t.turn_hit_tokens for t in traces)
    branch_count = sum(1 for t in traces if t.is_branch)
    new_branch_count = sum(1 for t in traces if t.is_new_branch)

    pr_rates = [t.per_request_token_hit_rate for t in traces]
    pr_sorted = sorted(pr_rates)

    page_hr = hit_pages / total_pages_needed if total_pages_needed else 0.0
    tok_hr = load_tokens / total_in if total_in else 0.0
    turn_hr = turn_hit_tokens / total_in if total_in else 0.0

    lcr_out: Optional[float] = None
    if compute_tokens > 0:
        lcr = load_tokens / compute_tokens
        if math.isfinite(lcr):
            lcr_out = float(lcr)

    # HBM tree histograms (primary tree).
    dh = hbm_tree.depth_histogram()
    vch = hbm_tree.valid_cached_depth_histogram()
    vad = hbm_tree.visit_counts_by_depth()

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

    # Tier-specific usage.
    hbm_u = state.hbm_usage_samples
    dram_u = state.dram_usage_samples
    hbm_peak = max(hbm_u) if hbm_u else 0
    hbm_avg = sum(hbm_u) / len(hbm_u) if hbm_u else 0.0
    dram_peak = max(dram_u) if dram_u else 0
    dram_avg = sum(dram_u) / len(dram_u) if dram_u else 0.0

    # Tier hit rates.
    hbm_hit_total = sum(t.hbm_hit_tokens for t in traces)
    dram_hit_total = sum(t.dram_hit_tokens for t in traces)
    hbm_hr = hbm_hit_total / total_in if total_in else 0.0
    dram_hr = dram_hit_total / total_in if total_in else 0.0

    # Flow totals.
    total_promoted_tokens = sum(t.promoted_tokens for t in traces)
    total_promoted_nodes = sum(t.promoted_nodes for t in traces)
    total_demoted_tokens = sum(t.demoted_tokens for t in traces)
    total_demoted_nodes = sum(t.demoted_nodes for t in traces)

    # GB conversion.
    avg_promoted_gb: Optional[float] = None
    avg_demoted_gb: Optional[float] = None
    if model is not None:
        bpt = model.kv_bytes_per_token
        if bpt > 0:
            avg_promoted_gb = (total_promoted_tokens / n_req) * bpt / (1024 ** 3)
            avg_demoted_gb = (total_demoted_tokens / n_req) * bpt / (1024 ** 3)

    # FLOPs save rate (uses effective combined hits).
    flops_save_rate: Optional[float] = None
    total_flops_no_cache: Optional[float] = None
    total_flops_with_cache: Optional[float] = None
    if model is not None:
        no_cache = sum(model.prefill_flops(t.input_tokens) for t in traces)
        with_cache = sum(
            model.incremental_prefill_flops(t.input_tokens, t.hit_tokens)
            for t in traces
        )
        total_flops_no_cache = no_cache
        total_flops_with_cache = with_cache
        flops_save_rate = 1.0 - with_cache / no_cache if no_cache > 0 else 0.0

    return MultiTierRunMetrics(
        page_level_hit_rate=page_hr,
        token_level_hit_rate=tok_hr,
        turn_level_hit_rate=turn_hr,
        per_request_hit_rate_mean=sum(pr_rates) / len(pr_rates),
        per_request_hit_rate_p50=_percentile(pr_sorted, 50),
        per_request_hit_rate_p90=_percentile(pr_sorted, 90),
        per_request_hit_rate_p99=_percentile(pr_sorted, 99),
        load_tokens=load_tokens,
        compute_tokens=compute_tokens,
        load_compute_ratio=lcr_out,
        peak_cached_tokens=hbm_peak + dram_peak,
        avg_cached_tokens=hbm_avg + dram_avg,
        num_requests=n_req,
        total_input_tokens=total_in,
        tree_depth_histogram=tree_depth_histogram,
        valid_cached_depth_histogram=valid_cached_depth_histogram,
        tree_access_by_depth=tree_access_by_depth,
        access_percentiles_by_depth=access_percentiles_by_depth,
        req_branch_rate=branch_count / n_req if n_req else 0.0,
        req_new_branch_rate=new_branch_count / n_req if n_req else 0.0,
        flops_save_rate=flops_save_rate,
        total_flops_no_cache=total_flops_no_cache,
        total_flops_with_cache=total_flops_with_cache,
        # Tier-specific fields.
        hbm_token_hit_rate=hbm_hr,
        dram_token_hit_rate=dram_hr,
        hbm_peak_cached_tokens=hbm_peak,
        hbm_avg_cached_tokens=hbm_avg,
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
