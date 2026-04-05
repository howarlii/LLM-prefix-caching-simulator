#!/usr/bin/env python
"""Run a single simulation with tree-mutation logging for the viewer.

Usage examples::

    python viz/run_with_log.py --dataset loogle --strategy lru --capacity 8GB
    python viz/run_with_log.py --dataset loogle --strategy crf_decoupling --page-size 64 --capacity 4GB
    python viz/run_with_log.py --dataset sharegpt --strategy marconi --max-requests 200

The log file is written to ``viz/logs/<params>.jsonl`` and can be loaded in
``viz/viewer.html``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root: ``python viz/run_with_log.py …``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.runner import (
    capacity_from_spec,
    effective_page_size,
    prepare_requests,
    run_simulation,
    strategy_from_name,
)
from src.config import DEFAULT_TOKENIZER_NAME, KV_BYTES_PER_TOKEN_DEFAULT
from viz.tree_logger import TreeLogger, _log_filename


def main() -> None:
    p = argparse.ArgumentParser(description="Run simulation with tree logging")
    p.add_argument("--dataset", required=True)
    p.add_argument("--strategy", default="lru")
    p.add_argument("--page-size", type=int, default=256)
    p.add_argument("--capacity", default="40GB")
    p.add_argument("--ordering", default="random")
    p.add_argument("--max-requests", type=int, default=1000)
    p.add_argument("--mamba-equiv", type=int, default=1000)
    p.add_argument(
        "--kv-bytes-per-token", type=int, default=KV_BYTES_PER_TOKEN_DEFAULT
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    p.add_argument("--out-dir", default="viz/logs")
    args = p.parse_args()

    strat = strategy_from_name(args.strategy)
    ps = effective_page_size(args.dataset, args.page_size)
    cap = capacity_from_spec(args.capacity, args.kv_bytes_per_token)

    print(f"Preparing requests ({args.dataset}, ordering={args.ordering}) …")
    reqs = prepare_requests(
        args.dataset,
        args.ordering,
        tokenizer_name=args.tokenizer,
        max_requests=args.max_requests,
    )
    if not reqs:
        print("No requests loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(reqs)} requests ready.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _log_filename(
        dataset=args.dataset,
        strategy=args.strategy,
        page_size=ps,
        capacity_spec=args.capacity,
        ordering=args.ordering,
        mamba_equiv=args.mamba_equiv,
    )
    log_path = out_dir / fname

    print(f"Running simulation (strategy={args.strategy}, ps={ps}, cap={args.capacity}) …")
    with TreeLogger(log_path) as logger:
        metrics = run_simulation(
            reqs,
            page_size=ps,
            strategy=strat,
            capacity_tokens=cap,
            mamba_state_token_equiv=args.mamba_equiv,
            logger=logger,
        )

    hr = metrics.get("token_level_hit_rate", 0) if isinstance(metrics, dict) else getattr(metrics, "token_level_hit_rate", 0)
    print(f"Done.  Token hit rate: {hr:.4f}")
    print(f"Log written to: {log_path}")
    print(f"Open viz/viewer.html in a browser and load this file.")


if __name__ == "__main__":
    main()
