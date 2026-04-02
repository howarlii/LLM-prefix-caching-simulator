"""Aggregate metrics from simulation traces and radix tree."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.cache_simulator import SimulationState
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_run_metrics(
    state: SimulationState,
    tree: RadixTree,
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
    )
