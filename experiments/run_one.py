#!/usr/bin/env python3
"""Run a single simulation with fully explicit parameters.

Two-tier (HBM + DRAM) cache is enabled by passing ``--dram-strategy`` *and* a
non-zero ``--dram-capacity``.  Otherwise the run is single-tier (HBM only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (
    DEFAULT_GPU_FLOPS,
    DEFAULT_PCIE_BANDWIDTH,
    DEFAULT_TOKENIZER_NAME,
    RESULTS_DIR,
)
from src.model_config import DEFAULT_MODEL, ModelConfig
from experiments.runner import (
    capacity_from_spec,
    effective_page_size,
    persist_result_row,
    prepare_requests,
    run_simulation,
    strategy_from_name,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a single KV cache simulation")
    p.add_argument("--dataset", default="swesmith",
                   help="loogle | narrativeqa | sharegpt | sharegpt_90k_raw | mooncake_toolagent | mooncake_conversation")
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument("--ordering", default="timestamp",
                   help="original | min_distance | max_distance | random | timestamp")
    p.add_argument("--sessions-per-second", type=float, default=1.0,
                   help="Session arrival rate (only used with --ordering timestamp)")
    p.add_argument("--words-per-min", type=float, default=90.0,
                   help="Simulated user typing speed (only used with --ordering timestamp)")
    p.add_argument("--strategy", default="marconi3_ev1_nt",
                   help="HBM eviction strategy: lru | lfu | fifo | marconi | ...")
    p.add_argument("--capacity", default="20",
                   help="HBM cache capacity in GB (e.g. 20) or inf")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenize-workers", type=int, default=0)
    p.add_argument("--max-requests", type=int, default=5000,
                   help="0 = full dataset")
    p.add_argument("--model", default=DEFAULT_MODEL.name,
                   help=f"Model name ({', '.join(ModelConfig.list_models())})")
    p.add_argument("--kv-bytes-per-token", type=int, default=0,
                   help="0 = use model default")
    # Hardware throughput (used by saved-time metrics)
    p.add_argument("--gpu-flops", type=float, default=DEFAULT_GPU_FLOPS,
                   help=f"GPU compute throughput in FLOP/sec (default {DEFAULT_GPU_FLOPS:.3g})")
    p.add_argument("--pcie-bandwidth", type=float, default=DEFAULT_PCIE_BANDWIDTH,
                   help=f"PCIe bandwidth in bytes/sec (default {DEFAULT_PCIE_BANDWIDTH:.3g})")
    # DRAM tier (set both to enable two-tier cache)
    p.add_argument("--dram-strategy", default=None,
                   help="DRAM eviction strategy. None disables the DRAM tier.")
    p.add_argument("--dram-capacity", default="0",
                   help="DRAM capacity in GB, or 'inf' for unlimited. 0 disables the DRAM tier.")

    p.add_argument("--out-csv", type=Path, default=None,
                   help="default: results/results.csv")
    p.add_argument("--out-json-dir", type=Path, default=None,
                   help="default: results/json/")
    args = p.parse_args()

    if args.out_csv is None:
        args.out_csv = RESULTS_DIR / f"results_{args.dataset}.csv"
    if args.out_json_dir is None:
        args.out_json_dir = RESULTS_DIR / f"json_{args.dataset}"

    model = ModelConfig.from_name(args.model)
    kv_b = args.kv_bytes_per_token or model.kv_bytes_per_token
    page_size = effective_page_size(args.dataset, args.page_size)
    cap = capacity_from_spec(args.capacity, kv_b)

    # DRAM tier — enabled iff a strategy is provided AND capacity != 0.
    # capacity_from_spec returns None for "inf"/"unlimited" → unlimited DRAM.
    dram_cap_tokens: int | None = 0
    dram_cap_spec = ""
    if args.dram_strategy:
        dram_cap_val = capacity_from_spec(args.dram_capacity, kv_b)
        if dram_cap_val is None or dram_cap_val > 0:
            dram_cap_tokens = dram_cap_val
            dram_cap_spec = args.dram_capacity

    reqs = prepare_requests(
        args.dataset,
        args.ordering,
        tokenizer_name=args.tokenizer,
        seed=args.seed,
        tokenize_workers=args.tokenize_workers,
        max_requests=args.max_requests or None,
        sessions_per_second=args.sessions_per_second,
        words_per_min=args.words_per_min,
    )
    if not reqs:
        raise SystemExit(f"No requests loaded for dataset {args.dataset!r}")

    # Build strategies AFTER loading requests so oracle strategies can
    # consume the full request sequence at construction time.
    dram_strat = None
    if args.dram_strategy and dram_cap_tokens != 0:
        dram_strat = strategy_from_name(
            args.dram_strategy,
            model=model,
            gpu_flops=args.gpu_flops,
            pcie_bandwidth=args.pcie_bandwidth,
            requests=reqs,
            page_size=page_size,
        )

    strat = strategy_from_name(
        args.strategy,
        model=model,
        gpu_flops=args.gpu_flops,
        pcie_bandwidth=args.pcie_bandwidth,
        requests=reqs,
        page_size=page_size,
    )
    metrics = run_simulation(
        reqs, page_size, strat, cap,
        dram_strategy=dram_strat,
        dram_capacity_tokens=dram_cap_tokens,
        model=model,
        gpu_flops=args.gpu_flops,
        pcie_bandwidth=args.pcie_bandwidth,
    )
    row = {
        "dataset": args.dataset,
        "page_size": page_size,
        "ordering": args.ordering,
        "strategy": args.strategy,
        "capacity_spec": args.capacity,
        "dram_strategy": args.dram_strategy or "",
        "dram_capacity_spec": dram_cap_spec,
        "tokenizer": args.tokenizer,
        "model_name": model.name,
        "gpu_flops": args.gpu_flops,
        "pcie_bandwidth": args.pcie_bandwidth,
        "metrics": metrics.to_dict(),
    }
    persist_result_row(args.out_csv, args.out_json_dir, row)

    flop_info = f" flop_save={metrics.flop_save_rate:.4f}" if metrics.flop_save_rate is not None else ""
    saved_info = (
        f" saved_time={metrics.total_saved_time:.3f}s "
        f"({metrics.saved_time_rate:.4f} of no-cache)"
        if metrics.total_saved_time is not None else ""
    )
    total_hr = metrics.hbm_token_hit_rate + metrics.dram_token_hit_rate
    if dram_strat is not None:
        print(
            f"dataset={args.dataset} page_size={page_size} ordering={args.ordering}\n"
            f"  hbm:{args.strategy}/{args.capacity} dram:{args.dram_strategy}/{args.dram_capacity} model={model.name}\n"
            f"  token_hr={total_hr:.4f} "
            f"hbm_hr={metrics.hbm_token_hit_rate:.4f} dram_hr={metrics.dram_token_hit_rate:.4f}\n"
            f"  promoted={metrics.avg_promoted_tokens_per_req:.1f} tok/req "
            f"({metrics.avg_promoted_nodes_per_req:.2f} nodes/req)  "
            f"demoted={metrics.avg_demoted_tokens_per_req:.1f} tok/req "
            f"({metrics.avg_demoted_nodes_per_req:.2f} nodes/req)\n"
            f" {flop_info}{saved_info}"
        )
    else:
        print(
            f"dataset={args.dataset} page_size={page_size} ordering={args.ordering} "
            f"strategy={args.strategy} capacity={args.capacity} model={model.name} "
            f"hbm_hr={metrics.hbm_token_hit_rate:.4f} "
            f"branch_rate={metrics.req_branch_rate:.4f} "
            f"new_branch_rate={metrics.req_new_branch_rate:.4f}"
            f"{flop_info}{saved_info}"
        )


if __name__ == "__main__":
    main()
