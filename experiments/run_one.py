#!/usr/bin/env python3
"""Run a single simulation with fully explicit parameters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DEFAULT_TOKENIZER_NAME, RESULTS_DIR
from src.model_config import DEFAULT_MODEL, ModelConfig
from experiments.runner import (
    capacity_from_spec,
    effective_page_size,
    persist_result_row,
    prepare_requests,
    run_multi_tier_simulation,
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
    p.add_argument("--strategy", default="marconi3_ev1_mn0",
                   help="lru | lfu | fifo | marconi")
    p.add_argument("--capacity", default="20",
                   help="Cache capacity in GB (e.g. 20) or inf")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenize-workers", type=int, default=0)
    p.add_argument("--max-requests", type=int, default=5000,
                   help="0 = full dataset")
    p.add_argument("--model", default=DEFAULT_MODEL.name,
                   help=f"Model name ({', '.join(ModelConfig.list_models())})")
    p.add_argument("--kv-bytes-per-token", type=int, default=0,
                   help="0 = use model default")
    # Multi-tier cache options
    p.add_argument("--multi-tier", action="store_true",
                   help="Enable two-tier (HBM + DRAM) cache simulation")
    p.add_argument("--hbm-capacity", default=None,
                   help="HBM capacity in GB (required with --multi-tier)")
    p.add_argument("--dram-capacity", default=None,
                   help="DRAM capacity in GB (required with --multi-tier)")
    p.add_argument("--hbm-strategy", default=None,
                   help="HBM eviction strategy (defaults to --strategy)")
    p.add_argument("--dram-strategy", default=None,
                   help="DRAM eviction strategy (defaults to --strategy)")

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

    if args.multi_tier:
        if args.hbm_capacity is None or args.dram_capacity is None:
            raise SystemExit("--hbm-capacity and --dram-capacity are required with --multi-tier")
        hbm_cap = capacity_from_spec(args.hbm_capacity, kv_b)
        dram_cap = capacity_from_spec(args.dram_capacity, kv_b)
        if hbm_cap is None or dram_cap is None:
            raise SystemExit("HBM and DRAM capacities must be finite for multi-tier mode")
        hbm_strat_name = args.hbm_strategy or args.strategy
        dram_strat_name = args.dram_strategy or args.strategy
        hbm_strat = strategy_from_name(hbm_strat_name, model=model)
        dram_strat = strategy_from_name(dram_strat_name, model=model)
        metrics = run_multi_tier_simulation(
            reqs, page_size, hbm_strat, dram_strat,
            hbm_capacity_tokens=hbm_cap,
            dram_capacity_tokens=dram_cap,
            model=model,
        )
        cap_spec = f"hbm{args.hbm_capacity}_dram{args.dram_capacity}"
        strat_spec = f"hbm:{hbm_strat_name}_dram:{dram_strat_name}"
        row = {
            "dataset": args.dataset,
            "page_size": page_size,
            "ordering": args.ordering,
            "strategy": strat_spec,
            "capacity_spec": cap_spec,
            "tokenizer": args.tokenizer,
            "mamba_state_token_equiv": model.mamba_state_token_equiv,
            "model_name": model.name,
            "metrics": metrics.to_dict(),
        }
        persist_result_row(args.out_csv, args.out_json_dir, row)
        flops_info = f" flops_save={metrics.flops_save_rate:.4f}" if metrics.flops_save_rate is not None else ""
        print(
            f"dataset={args.dataset} page_size={page_size} ordering={args.ordering} "
            f"hbm_strategy={hbm_strat_name} dram_strategy={dram_strat_name} "
            f"hbm_cap={args.hbm_capacity} dram_cap={args.dram_capacity} model={model.name}\n"
            f"  token_hr={metrics.token_level_hit_rate:.4f} "
            f"hbm_hr={metrics.hbm_token_hit_rate:.4f} dram_hr={metrics.dram_token_hit_rate:.4f} "
            f"branch_rate={metrics.req_branch_rate:.4f} "
            f"new_branch_rate={metrics.req_new_branch_rate:.4f}\n"
            f"  promoted={metrics.avg_promoted_tokens_per_req:.1f} tok/req "
            f"({metrics.avg_promoted_nodes_per_req:.2f} nodes/req"
            f"{f', {metrics.avg_promoted_gb_per_req:.6f} GB/req' if metrics.avg_promoted_gb_per_req is not None else ''})\n"
            f"  demoted={metrics.avg_demoted_tokens_per_req:.1f} tok/req "
            f"({metrics.avg_demoted_nodes_per_req:.2f} nodes/req"
            f"{f', {metrics.avg_demoted_gb_per_req:.6f} GB/req' if metrics.avg_demoted_gb_per_req is not None else ''})"
            f"{flops_info}"
        )
    else:
        strat = strategy_from_name(args.strategy, model=model)
        metrics = run_simulation(
            reqs, page_size, strat, cap,
            model=model,
        )
        row = {
            "dataset": args.dataset,
            "page_size": page_size,
            "ordering": args.ordering,
            "strategy": args.strategy,
            "capacity_spec": args.capacity,
            "tokenizer": args.tokenizer,
            "mamba_state_token_equiv": model.mamba_state_token_equiv,
            "model_name": model.name,
            "metrics": metrics.to_dict(),
        }
        persist_result_row(args.out_csv, args.out_json_dir, row)
        flops_info = f" flops_save={metrics.flops_save_rate:.4f}" if metrics.flops_save_rate is not None else ""
        print(
            f"dataset={args.dataset} page_size={page_size} ordering={args.ordering} "
            f"strategy={args.strategy} capacity={args.capacity} model={model.name} "
            f"token_hr={metrics.token_level_hit_rate:.4f} "
            f"page_hr={metrics.page_level_hit_rate:.4f} "
            f"turn_hr={metrics.turn_level_hit_rate:.4f} "
            f"branch_rate={metrics.req_branch_rate:.4f} "
            f"new_branch_rate={metrics.req_new_branch_rate:.4f}"
            f"{flops_info}"
        )


if __name__ == "__main__":
    main()
