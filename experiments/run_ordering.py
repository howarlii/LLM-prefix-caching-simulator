#!/usr/bin/env python3
"""Compare request orderings (cache distance)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DEFAULT_TOKENIZER_NAME, RESULTS_DIR
from experiments.runner import (
    capacity_from_spec,
    effective_page_size,
    persist_result_row,
    prepare_requests,
    run_simulation,
    strategy_from_name,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Request ordering experiment")
    p.add_argument("--dataset", default="loogle")
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument(
        "--orderings",
        default="original,min_distance,max_distance,random",
        help="Comma-separated ordering modes",
    )
    p.add_argument("--strategy", default="lru")
    p.add_argument("--capacity", default="inf")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenize-workers", type=int, default=0)
    p.add_argument("--max-requests", type=int, default=0, help="0 = use full dataset")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="default: results/ordering_sweep_<dataset>.csv",
    )
    p.add_argument(
        "--out-json-dir",
        type=Path,
        default=None,
        help="default: results/ordering_json_<dataset>/",
    )
    args = p.parse_args()

    if args.out_csv is None:
        args.out_csv = RESULTS_DIR / f"ordering_sweep_{args.dataset}.csv"
    if args.out_json_dir is None:
        args.out_json_dir = RESULTS_DIR / f"ordering_json_{args.dataset}"

    from src.config import KV_BYTES_PER_TOKEN_DEFAULT

    cap = capacity_from_spec(args.capacity, KV_BYTES_PER_TOKEN_DEFAULT)
    strat = strategy_from_name(args.strategy)
    page_size = effective_page_size(args.dataset, args.page_size)
    if page_size != args.page_size:
        print(
            f"Mooncake trace: overriding --page-size {args.page_size} -> {page_size}.",
            file=sys.stderr,
        )
    modes = [x.strip() for x in args.orderings.split(",") if x.strip()]

    for mode in modes:
        reqs = prepare_requests(
            args.dataset,
            mode,
            tokenizer_name=args.tokenizer,
            seed=args.seed,
            tokenize_workers=args.tokenize_workers,
            max_requests=args.max_requests or None,
        )
        if not reqs:
            raise SystemExit(f"No requests for {args.dataset!r}")
        metrics = run_simulation(reqs, page_size, strat, cap)
        row = {
            "experiment": "ordering",
            "dataset": args.dataset,
            "page_size": page_size,
            "ordering": mode,
            "strategy": args.strategy,
            "capacity_spec": args.capacity,
            "tokenizer": args.tokenizer,
            "metrics": metrics.to_dict(),
        }
        persist_result_row(args.out_csv, args.out_json_dir, row)
        print(f"ordering={mode} token_hr={metrics.token_level_hit_rate:.4f}")


if __name__ == "__main__":
    main()
