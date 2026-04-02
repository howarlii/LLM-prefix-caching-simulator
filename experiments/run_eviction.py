#!/usr/bin/env python3
"""Compare LRU / LFU / FIFO eviction under a fixed capacity."""

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
    p = argparse.ArgumentParser(description="Eviction policy comparison")
    p.add_argument("--dataset", default="loogle")
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument("--ordering", default="original")
    p.add_argument(
        "--strategies",
        default="lru,lfu,fifo",
        help="Comma-separated: lru, lfu, fifo",
    )
    p.add_argument(
        "--capacities",
        default="20,40,80,160",
        help="Comma-separated GB values (finite); add inf separately if needed",
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenize-workers", type=int, default=0)
    p.add_argument("--max-requests", type=int, default=0, help="0 = use full dataset")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="default: results/eviction_sweep_<dataset>.csv",
    )
    p.add_argument(
        "--out-json-dir",
        type=Path,
        default=None,
        help="default: results/eviction_json_<dataset>/",
    )
    args = p.parse_args()

    if args.out_csv is None:
        args.out_csv = RESULTS_DIR / f"eviction_sweep_{args.dataset}.csv"
    if args.out_json_dir is None:
        args.out_json_dir = RESULTS_DIR / f"eviction_json_{args.dataset}"

    from src.config import KV_BYTES_PER_TOKEN_DEFAULT

    reqs = prepare_requests(
        args.dataset,
        args.ordering,
        tokenizer_name=args.tokenizer,
        seed=args.seed,
        tokenize_workers=args.tokenize_workers,
        max_requests=args.max_requests or None,
    )
    if not reqs:
        raise SystemExit(f"No requests for {args.dataset!r}")

    page_size = effective_page_size(args.dataset, args.page_size)
    if page_size != args.page_size:
        print(
            f"Mooncake trace: overriding --page-size {args.page_size} -> {page_size}.",
            file=sys.stderr,
        )

    def norm_cap(s: str) -> str:
        s = s.strip().lower().replace("gb", "").strip()
        if s == "inf":
            return "inf"
        return s + "gb"

    capspecs = [norm_cap(x) for x in args.capacities.split(",") if x.strip()]

    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    for cap_s in capspecs:
        cap = capacity_from_spec(cap_s, KV_BYTES_PER_TOKEN_DEFAULT)
        for strat_name in strategies:
            strat = strategy_from_name(strat_name)
            metrics = run_simulation(reqs, page_size, strat, cap)
            row = {
                "experiment": "eviction",
                "dataset": args.dataset,
                "page_size": page_size,
                "ordering": args.ordering,
                "strategy": strat_name,
                "capacity_spec": cap_s,
                "tokenizer": args.tokenizer,
                "metrics": metrics.to_dict(),
            }
            persist_result_row(args.out_csv, args.out_json_dir, row)
            print(
                f"cap={cap_s} strat={strat_name} token_hr={metrics.token_level_hit_rate:.4f} "
                f"peak={metrics.peak_cached_tokens}"
            )


if __name__ == "__main__":
    main()
