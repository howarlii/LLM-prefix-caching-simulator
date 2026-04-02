#!/usr/bin/env python3
"""Scan page_size vs cache hit rate (and related metrics)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DEFAULT_TOKENIZER_NAME, RESULTS_DIR
from src.datasets_loader import is_mooncake_trace_dataset

from experiments.runner import (
    capacity_from_spec,
    effective_page_size,
    persist_result_row,
    prepare_requests,
    run_simulation,
    strategy_from_name,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Page size sweep experiment")
    p.add_argument(
        "--dataset",
        default="loogle",
        help="loogle | narrativeqa | sharegpt | mooncake_toolagent | mooncake_conversation "
        "(aliases: toolagent_trace, conversation_trace)",
    )
    p.add_argument(
        "--page-sizes",
        default="1,16,32,64,128,256,512,1024",
        help="Comma-separated page sizes",
    )
    p.add_argument("--ordering", default="original")
    p.add_argument("--strategy", default="lru")
    p.add_argument("--capacity", default="inf", help="GB (e.g. 20) or inf")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenize-workers", type=int, default=0)
    p.add_argument("--max-requests", type=int, default=0, help="0 = use full dataset")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="default: results/page_size_sweep_<dataset>.csv",
    )
    p.add_argument(
        "--out-json-dir",
        type=Path,
        default=None,
        help="default: results/page_size_json_<dataset>/",
    )
    p.add_argument("--kv-bytes-per-token", type=int, default=0, help="0 = use default from config")
    args = p.parse_args()

    if args.out_csv is None:
        args.out_csv = RESULTS_DIR / f"page_size_sweep_{args.dataset}.csv"
    if args.out_json_dir is None:
        args.out_json_dir = RESULTS_DIR / f"page_size_json_{args.dataset}"

    from src.config import KV_BYTES_PER_TOKEN_DEFAULT

    kv_b = args.kv_bytes_per_token or KV_BYTES_PER_TOKEN_DEFAULT
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
    sizes = [int(x.strip()) for x in args.page_sizes.split(",") if x.strip()]
    if is_mooncake_trace_dataset(args.dataset):
        bad = [s for s in sizes if s != 1]
        if bad:
            print(
                "Mooncake traces use one block hash per radix step; running only page_size=1 "
                f"(ignoring {bad!r}).",
                file=sys.stderr,
            )
        sizes = [1]

    for ps in sizes:
        ps = effective_page_size(args.dataset, ps)
        metrics = run_simulation(reqs, ps, strat, cap)
        row = {
            "experiment": "page_size",
            "dataset": args.dataset,
            "page_size": ps,
            "ordering": args.ordering,
            "strategy": args.strategy,
            "capacity_spec": args.capacity,
            "tokenizer": args.tokenizer,
            "metrics": metrics.to_dict(),
        }
        persist_result_row(args.out_csv, args.out_json_dir, row)
        print(
            f"page_size={ps} token_hr={metrics.token_level_hit_rate:.4f} "
            f"page_hr={metrics.page_level_hit_rate:.4f}"
        )


if __name__ == "__main__":
    main()
