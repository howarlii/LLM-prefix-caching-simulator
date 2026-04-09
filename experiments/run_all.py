#!/usr/bin/env python3
"""
Define and run multiple experiments concurrently.

Execution is split into two phases:
  1. **Prepare** — tokenize each unique (dataset, ordering) combo once (sequential).
  2. **Simulate** — run all simulations in a bounded ``multiprocessing.Pool``.

Edit the configuration section below to set up runs.

Two-tier (HBM + DRAM) caching is enabled per-experiment by setting both
``dram_strategy`` (non-empty) and ``dram_capacity`` (non-zero, finite).
Otherwise the run is single-tier (HBM only).
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
from viz.tree_logger import TreeLogger, _log_filename

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Global defaults (overridable per-experiment) ────────────────────────────
DATASET         = "swesmith" # swesmith | loogle | narrativeqa | sharegpt_90k_raw | mooncake_toolagent | mooncake_conversation
PAGE_SIZE       = 32
ORDERING        = "timestamp"  # original | min_distance | max_distance | random | timestamp
STRATEGY        = "marconi"
CAPACITY        = "160"          # GB, or "inf"/"unlimited" — HBM tier
MODEL_NAME      = DEFAULT_MODEL.name  # see ModelConfig.list_models() for options
TOKENIZER       = DEFAULT_TOKENIZER_NAME
SEED            = 0
MAX_REQUESTS    = 5000
TOKENIZE_WORKERS = 90

# ── Hardware throughput (used by saved-time metrics) ────────────────────────
GPU_FLOPS       = DEFAULT_GPU_FLOPS       # FLOP/sec; H100 BF16 dense
PCIE_BANDWIDTH  = DEFAULT_PCIE_BANDWIDTH  # bytes/sec; PCIe Gen5 x16

# ── Strategy-specific parameters ────────────────────────────────────────────
# These are passed to strategy constructors when strategy_from_name is called.
# Set to None to use the strategy's built-in default.
MARCONI_ALPHA    = None         # default: 1.5 for marconi, 1.5 for marconi2
CRF_LAMBDA_DECAY = None         # default: 0.5
CRF_C_ATTN       = None         # default: 1.0
CRF_C_SSM        = None         # default: 1.0

# ── DRAM tier defaults ──────────────────────────────────────────────────────
# Two-tier mode is enabled when DRAM_STRATEGY is non-empty AND DRAM_CAPACITY > 0.
# Per-experiment overrides via "dram_strategy" / "dram_capacity".
DRAM_STRATEGY    = None         # e.g. "marconi3_ev1_mn0" — None disables DRAM
DRAM_CAPACITY    = "0"          # GB (must be finite); "0" disables DRAM

# ── Logging (viz) ───────────────────────────────────────────────────────────
ENABLE_LOG      = False          # True = write tree-mutation JSONL per experiment
LOG_DIR         = "viz/logs"     # output directory for log files

# ── Parallelism ─────────────────────────────────────────────────────────────
MAX_WORKERS     = 100

# ── Experiment list ─────────────────────────────────────────────────────────
# Each dict is one simulation run.  Keys override the global defaults above.
# Strategy params can be set per-experiment via "marconi_alpha", "crf_lambda_decay", etc.
EXPERIMENTS: list[dict] = []

# Example: sweep page_size × capacity for marconi3 ablations
for ds in ["swesmith", "loogle", "narrativeqa", "sharegpt_90k_raw"]:
# for ds in ["swesmith"]:
    for page_size in [32]:
        for capacity in [20, 40, 80, 160, "inf"]:
        # for capacity in [20]:
            # EXPERIMENTS.append(dict(page_size=page_size, dataset=ds, strategy="marconi3_ev1_nt", capacity=capacity))
            EXPERIMENTS.append(dict(page_size=page_size, dataset=ds, strategy="branch_nt", capacity=capacity))

# Two-tier examples: HBM=branch_nt at small capacity, DRAM=marconi3_ev1_nt at large capacity.
for ds in ["swesmith", "loogle", "narrativeqa", "sharegpt_90k_raw"]:
    for page_size in [32]:
        for hbm_cap in [10, 20, 40, 80, "inf"]:
        # for hbm_cap in [40, 80, 160]:
            EXPERIMENTS.append(dict(
                page_size=page_size, dataset=ds,
                strategy="branch_nt", capacity=hbm_cap,
                dram_strategy="marconi3_ev1_nt", dram_capacity="inf",
            ))
            EXPERIMENTS.append(dict(
                page_size=page_size, dataset=ds,
                strategy="marconi3_ev1_nt", capacity=hbm_cap,
                dram_strategy="marconi3_ev1_nt", dram_capacity="inf",
            ))

# Log datasets
# ENABLE_LOG      = True
# MAX_REQUESTS    = 1000
# EXPERIMENTS: list[dict] = []
# for ds in ["swesmith", "loogle", "narrativeqa", "sharegpt_90k_raw"]:
#     for page_size in [1, 32]:
#         for capacity in [20, 80, "inf"]:
#             EXPERIMENTS.append(dict(page_size=page_size, strategy="marconi3_ev1_mn0",  capacity=capacity, dataset=ds))
#             EXPERIMENTS.append(dict(page_size=page_size, strategy="branch_nt",  capacity=capacity, dataset=ds))

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
        "model_name":             cfg.get("model_name", MODEL_NAME),
        "tokenizer":              cfg.get("tokenizer", TOKENIZER),
        "seed":                   cfg.get("seed", SEED),
        "max_requests":           cfg.get("max_requests", MAX_REQUESTS),
        "tokenize_workers":       cfg.get("tokenize_workers", TOKENIZE_WORKERS),
        # Strategy-specific params
        "marconi_alpha":          cfg.get("marconi_alpha", MARCONI_ALPHA),
        "crf_lambda_decay":       cfg.get("crf_lambda_decay", CRF_LAMBDA_DECAY),
        "crf_c_attn":             cfg.get("crf_c_attn", CRF_C_ATTN),
        "crf_c_ssm":              cfg.get("crf_c_ssm", CRF_C_SSM),
        # DRAM tier
        "dram_strategy":          cfg.get("dram_strategy", DRAM_STRATEGY),
        "dram_capacity":          cfg.get("dram_capacity", DRAM_CAPACITY),
        # Hardware throughput
        "gpu_flops":              cfg.get("gpu_flops", GPU_FLOPS),
        "pcie_bandwidth":         cfg.get("pcie_bandwidth", PCIE_BANDWIDTH),
    }


def _build_strategy_name(name: Optional[str], cfg: dict) -> Optional[str]:
    """Build the strategy string, embedding params as suffix when set.

    e.g. "marconi" with alpha=2.0 → "marconi_2.0"
         "crf_decoupling" with lambda=0.01 → "crf_decoupling_0.01"
    """
    if not name:
        return None
    n = name.lower()

    if n in ("marconi", "marconi2") and cfg.get("marconi_alpha") is not None:
        return f"{name}_{cfg['marconi_alpha']}"

    if n == "crf_decoupling" and cfg.get("crf_lambda_decay") is not None:
        return f"{name}_{cfg['crf_lambda_decay']}"

    return name


def _dram_enabled(cfg: dict) -> bool:
    return bool(cfg.get("dram_strategy")) and str(cfg.get("dram_capacity", "0")).strip() not in ("", "0")


def _label(cfg: dict) -> str:
    base = (
        f"{cfg.get('dataset','?')} "
        f"ps={cfg.get('page_size','?')} "
        f"{cfg.get('ordering','?')} "
        f"{cfg.get('strategy','?')} "
        f"cap={cfg.get('capacity','?')}"
    )
    if _dram_enabled(cfg):
        base += f" dram:{cfg['dram_strategy']}/{cfg['dram_capacity']}"
    return base


# ── Populated by main() before forking — child processes see it via COW ──────
_PREPARED: Dict[tuple, list] = {}


def _sim_worker(args: Tuple) -> Dict[str, Any]:
    """Run one simulation inside a pool worker (fork inherits _PREPARED)."""
    prep_key, page_size, sim_spec, cfg = args
    reqs = _PREPARED[prep_key]
    model = ModelConfig.from_name(cfg["model_name"])

    strategy = strategy_from_name(
        sim_spec["strategy_name"],
        model=model,
        gpu_flops=cfg["gpu_flops"],
        pcie_bandwidth=cfg["pcie_bandwidth"],
    )
    dram_strategy = None
    if sim_spec.get("dram_strategy_name"):
        dram_strategy = strategy_from_name(
            sim_spec["dram_strategy_name"],
            model=model,
            gpu_flops=cfg["gpu_flops"],
            pcie_bandwidth=cfg["pcie_bandwidth"],
        )

    logger = None
    if ENABLE_LOG:
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        fname = _log_filename(
            dataset=cfg["dataset"],
            strategy=sim_spec["strategy_name"],
            page_size=page_size,
            capacity_spec=str(cfg["capacity"]),
            ordering=cfg["ordering"],
            model_name=model.name,
        )
        logger = TreeLogger(log_dir / fname)

    try:
        metrics = run_simulation(
            reqs, page_size, strategy, sim_spec["cap"],
            dram_strategy=dram_strategy,
            dram_capacity_tokens=sim_spec.get("dram_cap", 0),
            logger=logger,
            model=model,
            gpu_flops=cfg["gpu_flops"],
            pcie_bandwidth=cfg["pcie_bandwidth"],
        )
    finally:
        if logger is not None:
            logger.close()

    return {"cfg": cfg, "metrics": metrics.to_dict()}


def main() -> None:
    global _PREPARED

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
            model = ModelConfig.from_name(cfg["model_name"])
            page_size = effective_page_size(cfg["dataset"], int(cfg["page_size"]))
            cap = capacity_from_spec(str(cfg["capacity"]), model=model)
            sim_spec: Dict[str, Any] = {
                "strategy_name": _build_strategy_name(cfg["strategy"], cfg),
                "cap": cap,
            }
            if _dram_enabled(cfg):
                # capacity_from_spec returns None for "inf" → unlimited DRAM.
                dram_cap = capacity_from_spec(str(cfg["dram_capacity"]), model=model)
                sim_spec["dram_strategy_name"] = _build_strategy_name(cfg["dram_strategy"], cfg)
                sim_spec["dram_cap"] = dram_cap
            sim_args.append((prep_key, page_size, sim_spec, cfg))

    print(
        f"\n=== Phase 2: Running {len(sim_args)} simulation(s) "
        f"(max_workers={MAX_WORKERS}) ==="
    )
    t_start = time.perf_counter()

    ctx = mp.get_context("fork")
    failed: List[str] = []
    done = 0

    with ctx.Pool(processes=min(MAX_WORKERS, len(EXPERIMENTS))) as pool:
        for result in pool.imap_unordered(_sim_worker, sim_args):
            cfg = result["cfg"]
            metrics_dict = result["metrics"]
            label = _label(cfg)

            dataset = cfg["dataset"]
            out_csv = RESULTS_DIR / f"results_{dataset}.csv"
            out_json_dir = RESULTS_DIR / f"json_{dataset}"

            strategy_label = _build_strategy_name(cfg["strategy"], cfg)
            dram_enabled = _dram_enabled(cfg)
            dram_strategy_label = (
                _build_strategy_name(cfg["dram_strategy"], cfg) or "" if dram_enabled else ""
            )
            dram_capacity_label = str(cfg["dram_capacity"]) if dram_enabled else ""
            row = {
                "dataset": dataset,
                "page_size": effective_page_size(dataset, int(cfg["page_size"])),
                "ordering": cfg["ordering"],
                "strategy": strategy_label,
                "capacity_spec": str(cfg["capacity"]),
                "dram_strategy": dram_strategy_label,
                "dram_capacity_spec": dram_capacity_label,
                "tokenizer": cfg["tokenizer"],
                "model_name": cfg["model_name"],
                "gpu_flops": cfg["gpu_flops"],
                "pcie_bandwidth": cfg["pcie_bandwidth"],
                "metrics": metrics_dict,
            }
            persist_result_row(out_csv, out_json_dir, row)

            done += 1
            hbm_hr = metrics_dict.get("hbm_token_hit_rate", 0)
            dram_hr = metrics_dict.get("dram_token_hit_rate", 0)
            total_hr = hbm_hr + dram_hr
            branch_r = metrics_dict.get("req_branch_rate", 0)
            new_branch_r = metrics_dict.get("req_new_branch_rate", 0)
            flop_sr = metrics_dict.get("flop_save_rate")
            flop_str = f"  flop_save={flop_sr:.4f}" if flop_sr is not None else ""
            saved_t = metrics_dict.get("total_saved_time")
            saved_str = f"  saved_time={saved_t:.2f}s" if saved_t is not None else ""
            if dram_enabled:
                tier_str = f"  hbm_hr={hbm_hr:.4f}  dram_hr={dram_hr:.4f}"
            else:
                tier_str = ""
            print(
                f"  [{done}/{len(sim_args)}] {label}  "
                f"token_hr={total_hr:.4f}{tier_str}  branch_rate={branch_r:.4f}  new_branch_rate={new_branch_r:.4f}{flop_str}{saved_str}",
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
