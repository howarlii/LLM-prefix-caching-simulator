#!/usr/bin/env python3
"""Run a single simulation with fully explicit parameters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DEFAULT_TOKENIZER_NAME, KV_BYTES_PER_TOKEN_DEFAULT, RESULTS_DIR
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
    p.add_argument("--dataset", default="loogle",
                   help="loogle | narrativeqa | sharegpt | sharegpt_90k_raw | mooncake_toolagent | mooncake_conversation")
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument("--ordering", default="random",
                   help="original | min_distance | max_distance | random")
    p.add_argument("--strategy", default="lru",
                   help="lru | lfu | fifo | marconi")
    p.add_argument("--capacity", default="160",
                   help="Cache capacity in GB (e.g. 20) or inf")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenize-workers", type=int, default=0)
    p.add_argument("--max-requests", type=int, default=5000,
                   help="0 = full dataset")
    p.add_argument("--mamba-state-token-equiv", type=int, default=1000,
                   help="Token-equivalent cost of one Mamba SSM state (0 = pure attention mode)")
    p.add_argument("--kv-bytes-per-token", type=int, default=0,
                   help="0 = use default from config")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="default: results/results.csv")
    p.add_argument("--out-json-dir", type=Path, default=None,
                   help="default: results/json/")
    args = p.parse_args()

    if args.out_csv is None:
        args.out_csv = RESULTS_DIR / f"results_{args.dataset}.csv"
    if args.out_json_dir is None:
        args.out_json_dir = RESULTS_DIR / f"json_{args.dataset}"

    kv_b = args.kv_bytes_per_token or KV_BYTES_PER_TOKEN_DEFAULT
    page_size = effective_page_size(args.dataset, args.page_size)
    cap = capacity_from_spec(args.capacity, kv_b)

    reqs = prepare_requests(
        args.dataset,
        args.ordering,
        tokenizer_name=args.tokenizer,
        seed=args.seed,
        tokenize_workers=args.tokenize_workers,
        max_requests=args.max_requests or None,
    )
    if not reqs:
        raise SystemExit(f"No requests loaded for dataset {args.dataset!r}")

    strat = strategy_from_name(args.strategy)
    metrics = run_simulation(
        reqs, page_size, strat, cap,
        mamba_state_token_equiv=args.mamba_state_token_equiv,
    )

    row = {
        "dataset": args.dataset,
        "page_size": page_size,
        "ordering": args.ordering,
        "strategy": args.strategy,
        "capacity_spec": args.capacity,
        "tokenizer": args.tokenizer,
        "mamba_state_token_equiv": args.mamba_state_token_equiv,
        "metrics": metrics.to_dict(),
    }
    persist_result_row(args.out_csv, args.out_json_dir, row)
    print(
        f"dataset={args.dataset} page_size={page_size} ordering={args.ordering} "
        f"strategy={args.strategy} capacity={args.capacity} "
        f"token_hr={metrics.token_level_hit_rate:.4f} "
        f"page_hr={metrics.page_level_hit_rate:.4f} "
        f"turn_hr={metrics.turn_level_hit_rate:.4f}"
    )


if __name__ == "__main__":
    main()
