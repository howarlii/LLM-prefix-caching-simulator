#!/usr/bin/env python3
"""
Define and run multiple experiments concurrently.

Execution is split into two phases:
  1. **Prepare** — tokenize each unique (dataset, ordering) combo once (sequential).
  2. **Simulate** — run all simulations in a bounded ``multiprocessing.Pool``.

Edit the configuration section below to set up runs.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from viz.tree_logger import TreeLogger, _log_filename

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Global defaults (overridable per-experiment) ────────────────────────────
DATASET         = "swesmith" # swesmith | loogle | narrativeqa | sharegpt_90k_raw | mooncake_toolagent | mooncake_conversation
PAGE_SIZE       = 32
ORDERING        = "random"
STRATEGY        = "marconi"
CAPACITY        = "160"          # GB, or "inf"/"unlimited"
MAMBA_STATE_TOKEN_EQUIV = 1000   # 0 = pure full-attention (no mamba)
TOKENIZER       = DEFAULT_TOKENIZER_NAME
SEED            = 0
MAX_REQUESTS    = 5000
TOKENIZE_WORKERS = 90

# ── Strategy-specific parameters ────────────────────────────────────────────
# These are passed to strategy constructors when strategy_from_name is called.
# Set to None to use the strategy's built-in default.
#
# Marconi / Marconi2:
MARCONI_ALPHA   = None           # default: 1.5 for marconi, 1.5 for marconi2
#
# CRF Decoupling:
CRF_LAMBDA_DECAY = None         # default: 0.5
CRF_C_ATTN       = None         # default: 1.0
CRF_C_SSM        = None         # default: 1.0

# ── Logging (viz) ───────────────────────────────────────────────────────────
ENABLE_LOG      = False          # True = write tree-mutation JSONL per experiment
LOG_DIR         = "viz/logs"     # output directory for log files

# ── Parallelism ─────────────────────────────────────────────────────────────
MAX_WORKERS     = 100

# ── Experiment list ─────────────────────────────────────────────────────────
# Each dict is one simulation run.  Keys override the global defaults above.
# Strategy params can be set per-experiment via "marconi_alpha", "crf_lambda_decay", etc.
#
# Shorthand: omit any key to use the global default.
EXPERIMENTS: list[dict] = []

# Example: sweep page_size × capacity for marconi and marconi2
for ds in ["swesmith", "loogle", "narrativeqa", "sharegpt_90k_raw"]:
# for ds in ["swesmith"]:
    for page_size in [1, 32, 256, 1024]:
    # for page_size in [1, 256]:
        for capacity in [80, 160, 320, 640, "inf"]:
            # EXPERIMENTS.append(dict(page_size=page_size, dataset=ds, strategy="lru",  capacity=capacity))
            # EXPERIMENTS.append(dict(page_size=page_size, dataset=ds, strategy="marconi",  capacity=capacity))
            # EXPERIMENTS.append(dict(page_size=page_size, dataset=ds, strategy="marconi3_ev0_mn0", capacity=capacity))
            # EXPERIMENTS.append(dict(page_size=page_size, dataset=ds, strategy="marconi3", capacity=capacity))
            # EXPERIMENTS.append(dict(page_size=page_size, dataset=ds, strategy="marconi3_ev1_mn0", capacity=capacity))
            # EXPERIMENTS.append(dict(page_size=page_size, dataset=ds, strategy="marconi3_ev1_mn1", capacity=capacity))
            EXPERIMENTS.append(dict(page_size=page_size, dataset=ds, strategy="marconi3_ev2_mn0", capacity=capacity))

# Log datasets
# ENABLE_LOG      = True
# MAX_REQUESTS    = 1000
# capacity = "inf"
# EXPERIMENTS: list[dict] = []
# for ds in ["swesmith", "loogle", "narrativeqa", "sharegpt_90k_raw"]:
#     # for page_size in [1, 32, 256]:
#     for page_size in [1]:
#         EXPERIMENTS.append(dict(page_size=page_size, strategy="marconi",  capacity=capacity, dataset=ds))
#         EXPERIMENTS.append(dict(page_size=page_size, strategy="marconi2",  capacity=capacity, dataset=ds))

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       END OF CONFIGURATION                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def _merge_defaults(cfg: dict) -> dict:
    """Fill in global defaults for any keys not set in the experiment dict."""
    return {
        "dataset":                cfg.get("dataset", DATASET),
        "page_size":              cfg.get("page_size", PAGE_SIZE),
        "ordering":               cfg.get("ordering", ORDERING),
        "strategy":               cfg.get("strategy", STRATEGY),
        "capacity":               cfg.get("capacity", CAPACITY),
        "mamba_state_token_equiv": cfg.get("mamba_state_token_equiv", MAMBA_STATE_TOKEN_EQUIV),
        "tokenizer":              cfg.get("tokenizer", TOKENIZER),
        "seed":                   cfg.get("seed", SEED),
        "max_requests":           cfg.get("max_requests", MAX_REQUESTS),
        "tokenize_workers":       cfg.get("tokenize_workers", TOKENIZE_WORKERS),
        # Strategy-specific params
        "marconi_alpha":          cfg.get("marconi_alpha", MARCONI_ALPHA),
        "crf_lambda_decay":       cfg.get("crf_lambda_decay", CRF_LAMBDA_DECAY),
        "crf_c_attn":             cfg.get("crf_c_attn", CRF_C_ATTN),
        "crf_c_ssm":              cfg.get("crf_c_ssm", CRF_C_SSM),
    }


def _build_strategy_name(cfg: dict) -> str:
    """Build the strategy string, embedding params as suffix when set.

    e.g. "marconi" with alpha=2.0 → "marconi_2.0"
         "crf_decoupling" with lambda=0.01 → "crf_decoupling_0.01"
    """
    name = cfg["strategy"]
    n = name.lower()

    if n in ("marconi", "marconi2") and cfg.get("marconi_alpha") is not None:
        return f"{name}_{cfg['marconi_alpha']}"

    if n == "crf_decoupling" and cfg.get("crf_lambda_decay") is not None:
        return f"{name}_{cfg['crf_lambda_decay']}"

    return name


def _label(cfg: dict) -> str:
    return (
        f"{cfg.get('dataset','?')} "
        f"ps={cfg.get('page_size','?')} "
        f"{cfg.get('ordering','?')} "
        f"{cfg.get('strategy','?')} "
        f"cap={cfg.get('capacity','?')}"
    )


# ── Populated by main() before forking — child processes see it via COW ──────
_PREPARED: Dict[tuple, list] = {}


def _sim_worker(args: Tuple) -> Dict[str, Any]:
    """Run one simulation inside a pool worker (fork inherits _PREPARED)."""
    prep_key, page_size, strategy_name, cap, mamba, cfg = args
    reqs = _PREPARED[prep_key]
    strategy = strategy_from_name(strategy_name)

    logger = None
    if ENABLE_LOG:
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        fname = _log_filename(
            dataset=cfg["dataset"],
            strategy=strategy_name,
            page_size=page_size,
            capacity_spec=str(cfg["capacity"]),
            ordering=cfg["ordering"],
            mamba_equiv=mamba,
        )
        logger = TreeLogger(log_dir / fname)

    try:
        metrics = run_simulation(reqs, page_size, strategy, cap, mamba, logger=logger)
    finally:
        if logger is not None:
            logger.close()

    return {"cfg": cfg, "metrics": metrics.to_dict()}


def main() -> None:
    global _PREPARED

    kv_bytes = KV_BYTES_PER_TOKEN_DEFAULT

    # Merge defaults into every experiment
    merged_experiments = [_merge_defaults(cfg) for cfg in EXPERIMENTS]

    # ── Phase 1: group experiments by prepare_requests inputs ────────────
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for cfg in merged_experiments:
        max_req = int(cfg.get("max_requests", 0)) or None
        prep_key = (
            cfg["dataset"],
            cfg["ordering"],
            cfg["tokenizer"],
            max_req,
            int(cfg["seed"]),
        )
        groups[prep_key].append(cfg)

    print(
        f"=== Phase 1: Preparing {len(groups)} unique dataset/ordering combo(s) "
        f"for {len(EXPERIMENTS)} experiments ==="
    )
    for prep_key in groups:
        dataset, ordering, tokenizer, max_requests, seed = prep_key
        label = f"{dataset}/{ordering}" + (f" (n={max_requests})" if max_requests else "")
        t0 = time.perf_counter()
        print(f"  Preparing: {label} ...", flush=True)
        reqs = prepare_requests(
            dataset,
            ordering,
            tokenizer,
            seed=seed,
            tokenize_workers=TOKENIZE_WORKERS,
            max_requests=max_requests,
        )
        elapsed = time.perf_counter() - t0
        _PREPARED[prep_key] = reqs
        print(f"  -> {len(reqs)} requests ready ({elapsed:.1f}s)")

    # ── Phase 2: build simulation arg list ───────────────────────────────
    sim_args: List[tuple] = []
    for prep_key, cfgs in groups.items():
        for cfg in cfgs:
            page_size = effective_page_size(cfg["dataset"], int(cfg["page_size"]))
            cap = capacity_from_spec(str(cfg["capacity"]), kv_bytes)
            mamba = int(cfg["mamba_state_token_equiv"])
            strategy_name = _build_strategy_name(cfg)
            sim_args.append((prep_key, page_size, strategy_name, cap, mamba, cfg))

    print(
        f"\n=== Phase 2: Running {len(sim_args)} simulation(s) "
        f"(max_workers={MAX_WORKERS}) ==="
    )
    t_start = time.perf_counter()

    ctx = mp.get_context("fork")
    failed: List[str] = []
    done = 0

    with ctx.Pool(processes=MAX_WORKERS) as pool:
        for result in pool.imap_unordered(_sim_worker, sim_args):
            cfg = result["cfg"]
            metrics_dict = result["metrics"]
            label = _label(cfg)

            dataset = cfg["dataset"]
            out_csv = RESULTS_DIR / f"results_{dataset}.csv"
            out_json_dir = RESULTS_DIR / f"json_{dataset}"

            row = {
                "dataset": dataset,
                "page_size": effective_page_size(dataset, int(cfg["page_size"])),
                "ordering": cfg["ordering"],
                "strategy": _build_strategy_name(cfg),
                "capacity_spec": str(cfg["capacity"]),
                "tokenizer": cfg["tokenizer"],
                "mamba_state_token_equiv": int(cfg["mamba_state_token_equiv"]),
                "metrics": metrics_dict,
            }
            persist_result_row(out_csv, out_json_dir, row)

            done += 1
            token_hr = metrics_dict.get("token_level_hit_rate", 0)
            print(
                f"  [{done}/{len(sim_args)}] {label}  "
                f"token_hr={token_hr:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - t_start

    if failed:
        print(f"\n{len(failed)} experiment(s) failed:", file=sys.stderr)
        for label in failed:
            print(f"  {label}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(EXPERIMENTS)} experiments completed in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
