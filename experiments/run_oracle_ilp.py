#!/usr/bin/env python3
"""Solve the offline-optimal cache scheduling ILP for a request sequence.

This is the **ground-truth** benchmark for the prefix-cache simulator:
given the full request stream and a fixed (HBM, DRAM) capacity it
returns the provably optimal ``saved_time_rate`` (within the linear
saved-time model — see ``src/strategies/oracle_ilp.py``).

The ILP is **slow**.  As a rule of thumb:

* ≤  100 requests : seconds.
* ≤  300 requests : minutes.
* ≤  500 requests : tens of minutes (sometimes hours; pass
  ``--time-limit`` to fall back to the best feasible incumbent).
* > 500 requests  : usually intractable — use ``oracle_greedy`` instead.

Pair this with ``oracle_greedy`` (run via ``run_one.py
--strategy oracle_greedy``) to see how close the heuristic gets to the
true optimum on the same prefix.
"""

from __future__ import annotations

import argparse
import json
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
from src.strategies.oracle_ilp import OracleILPSolver
from experiments.runner import (
    capacity_from_spec,
    effective_page_size,
    prepare_requests,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline-optimal ILP solver for the KV cache simulator"
    )
    p.add_argument("--dataset", default="loogle")
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument("--ordering", default="original",
                   help="original | min_distance | max_distance | random | timestamp")
    p.add_argument("--capacity", default="2",
                   help="HBM capacity in GB (e.g. 2) or inf")
    p.add_argument("--dram-capacity", default="0",
                   help="DRAM capacity in GB; 0 disables the DRAM tier")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenize-workers", type=int, default=0)
    p.add_argument("--max-requests", type=int, default=200,
                   help="Cap the request count fed to the ILP (default 200; "
                        "300+ is borderline, 500+ is usually intractable)")
    p.add_argument("--model", default=DEFAULT_MODEL.name)
    p.add_argument("--gpu-flops", type=float, default=DEFAULT_GPU_FLOPS)
    p.add_argument("--pcie-bandwidth", type=float, default=DEFAULT_PCIE_BANDWIDTH)
    p.add_argument("--time-limit", type=int, default=600,
                   help="CBC solver time limit in seconds (default 600)")
    p.add_argument("--relax", action="store_true",
                   help="Solve the LP relaxation (continuous in [0,1]) for an "
                        "upper bound — much faster but not always achievable")
    p.add_argument("--msg", action="store_true", help="Print solver progress")
    p.add_argument("--out-json", type=Path, default=None,
                   help="default: results/oracle_ilp_<dataset>.json")
    p.add_argument(
        "--sessions-per-second",
        type=float,
        default=1.0,
        help="Used only with --ordering timestamp",
    )
    p.add_argument(
        "--words-per-min",
        type=float,
        default=90.0,
        help="Used only with --ordering timestamp",
    )
    args = p.parse_args()

    model = ModelConfig.from_name(args.model)
    page_size = effective_page_size(args.dataset, args.page_size)
    cap_hbm = capacity_from_spec(args.capacity, model=model)
    if cap_hbm is None:
        raise SystemExit("--capacity must be a finite GB value (got 'inf')")
    if args.dram_capacity in ("0", "", None):
        cap_dram = 0
    else:
        cap_dram_val = capacity_from_spec(args.dram_capacity, model=model)
        cap_dram = cap_dram_val or 0

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

    print(
        f"Building ILP for dataset={args.dataset} num_requests={len(reqs)} "
        f"page_size={page_size} capacity={args.capacity}GB "
        f"dram={args.dram_capacity}GB model={model.name}"
    )
    solver = OracleILPSolver(
        requests_token_ids=[r.token_ids for r in reqs],
        page_size=page_size,
        capacity_hbm_tokens=cap_hbm,
        dram_capacity_tokens=cap_dram,
        model=model,
        gpu_flops=args.gpu_flops,
        pcie_bandwidth=args.pcie_bandwidth,
    )
    print(
        f"  global radix tree: {solver.N} nodes, "
        f"{len(solver.active_nodes)} reused (≥2 visits)"
    )

    result = solver.solve(time_limit_s=args.time_limit, msg=args.msg, relax=args.relax)
    print(
        f"\nstatus={result.status}  vars={result.num_binary_vars}  "
        f"constraints={result.num_constraints}  solve_time={result.solve_time_s:.2f}s"
    )
    print(
        f"saved_time_rate = {result.saved_time_rate:.6f}  "
        f"flop_save_rate = {result.flop_save_rate:.6f}"
    )
    print(
        f"  hbm_token_hit_rate = {result.hbm_token_hit_rate:.6f}  "
        f"dram_token_hit_rate = {result.dram_token_hit_rate:.6f}"
    )
    print(
        f"  total_saved_time = {result.total_saved_time:.4f}s  "
        f"(no-cache compute = {result.gpu_compute_time_no_cache:.4f}s)"
    )

    out = args.out_json or (RESULTS_DIR / f"oracle_ilp_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "ordering": args.ordering,
                "page_size": page_size,
                "capacity_spec": args.capacity,
                "dram_capacity_spec": args.dram_capacity,
                "model_name": model.name,
                "num_requests": len(reqs),
                "max_requests": args.max_requests,
                "relax": args.relax,
                "result": result.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
