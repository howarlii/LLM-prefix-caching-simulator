#!/usr/bin/env python3
"""
Define and run multiple experiments concurrently.
Edit EXPERIMENTS and DEFAULTS below to configure which runs to execute.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RUN_ONE = _ROOT / "experiments" / "run_one.py"

# ── Edit this to configure your experiments ───────────────────────────────────
# Each dict is one call to run_one.py. Keys map directly to CLI flags
# (underscores → hyphens). Omitted keys fall back to DEFAULTS or run_one defaults.
DATASET_NAMES = "sharegpt_90k_raw"
EXPERIMENTS: list[dict] = [
    # Page-size sweep
    # dict(dataset=DATASET_NAMES, page_size=2,   ordering="original", strategy="marconi", mamba_state_token_equiv="1000", capacity="inf"),
    # dict(dataset=DATASET_NAMES, page_size=8,   ordering="original", strategy="marconi", mamba_state_token_equiv="1000", capacity="inf"),
    # dict(dataset=DATASET_NAMES, page_size=32,  ordering="original", strategy="marconi", mamba_state_token_equiv="1000", capacity="inf"),
    # dict(dataset=DATASET_NAMES, page_size=128, ordering="original", strategy="marconi", mamba_state_token_equiv="1000", capacity="inf"),
    # dict(dataset=DATASET_NAMES, page_size=2,   ordering="original", strategy="lru", mamba_state_token_equiv="1000", capacity="inf"),
    # dict(dataset=DATASET_NAMES, page_size=8,   ordering="original", strategy="lru", mamba_state_token_equiv="1000", capacity="inf"),
    # dict(dataset=DATASET_NAMES, page_size=32,  ordering="original", strategy="lru", mamba_state_token_equiv="1000", capacity="inf"),
    # dict(dataset=DATASET_NAMES, page_size=128, ordering="original", strategy="lru", mamba_state_token_equiv="1000", capacity="inf"),
    # Ordering sweep
    # dict(dataset="loogle", page_size=32, ordering="min_distance", strategy="lru", capacity="inf"),
    # dict(dataset="loogle", page_size=32, ordering="max_distance", strategy="lru", capacity="inf"),
    # dict(dataset="loogle", page_size=32, ordering="random",       strategy="lru", capacity="inf"),
    # # Eviction sweep
    # dict(dataset="loogle", page_size=32, ordering="original", strategy="lru",  capacity="20"),
    # dict(dataset="loogle", page_size=32, ordering="original", strategy="lfu",  capacity="20"),
    # dict(dataset="loogle", page_size=32, ordering="original", strategy="fifo", capacity="20"),
]

DATASET_NAMES = "sharegpt_90k_raw"
DATASET_NAMES = "swesmith"
# DATASET_NAMES = "loogle"
for page_size in [32, 256, 1024]:
    for capacity in [20, 40, 80, 160, 320, 'inf']:
        # EXPERIMENTS.append(dict(dataset=DATASET_NAMES, page_size=page_size, ordering="original", strategy="marconi", mamba_state_token_equiv="1000", capacity=capacity))
        # EXPERIMENTS.append(dict(dataset=DATASET_NAMES, page_size=page_size, ordering="original", strategy="lru", mamba_state_token_equiv="1000", capacity=capacity))
        EXPERIMENTS.append(dict(dataset=DATASET_NAMES, page_size=page_size, ordering="random", strategy="marconi", mamba_state_token_equiv="1000", capacity=capacity))
        EXPERIMENTS.append(dict(dataset=DATASET_NAMES, page_size=page_size, ordering="random", strategy="lru", mamba_state_token_equiv="1000", capacity=capacity))

# DATASET_NAMES = "mooncake_toolagent"
# for page_size in [1, 8, 32]:
#     for capacity in [20, 40, 80, 160, 'inf']:
#         # EXPERIMENTS.append(dict(dataset=DATASET_NAMES, page_size=page_size, ordering="original", strategy="marconi", mamba_state_token_equiv="1000", capacity=capacity))
#         # EXPERIMENTS.append(dict(dataset=DATASET_NAMES, page_size=page_size, ordering="original", strategy="lru", mamba_state_token_equiv="1000", capacity=capacity))
#         EXPERIMENTS.append(dict(dataset=DATASET_NAMES, page_size=page_size, ordering="random", strategy="marconi", mamba_state_token_equiv="1000", capacity=capacity))
#         EXPERIMENTS.append(dict(dataset=DATASET_NAMES, page_size=page_size, ordering="random", strategy="lru", mamba_state_token_equiv="1000", capacity=capacity))

# ── Applied to every experiment unless the experiment dict overrides them ─────
DEFAULTS: dict = dict(
    max_requests=5000,
    tokenize_workers=90,
)

# ── How many experiments to run in parallel ───────────────────────────────────
MAX_WORKERS = 1
# ─────────────────────────────────────────────────────────────────────────────


_print_lock = threading.Lock()


def _label(cfg: dict) -> str:
    return (
        f"{cfg.get('dataset','?')} "
        f"ps={cfg.get('page_size','?')} "
        f"{cfg.get('ordering','?')} "
        f"{cfg.get('strategy','?')} "
        f"cap={cfg.get('capacity','?')}"
    )


def _build_cmd(cfg: dict) -> list[str]:
    merged = {**DEFAULTS, **cfg}
    cmd = [sys.executable, "-u", str(_RUN_ONE)]
    for k, v in merged.items():
        if v is None:
            continue
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd


def _run(cfg: dict, cmd: list[str]) -> int:
    label = _label(cfg)
    with _print_lock:
        print(f"[{label}] starting", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(_ROOT),
    )
    for line in proc.stdout:
        with _print_lock:
            print(f"[{label}] {line}", end="", flush=True)
    proc.wait()
    return proc.returncode


def main() -> None:
    jobs = [(cfg, _build_cmd(cfg)) for cfg in EXPERIMENTS]
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run, cfg, cmd): cfg for cfg, cmd in jobs}
        for fut in as_completed(futures):
            cfg = futures[fut]
            rc = fut.result()
            if rc != 0:
                failed.append(_label(cfg))

    if failed:
        print(f"\n{len(failed)} experiment(s) failed:", file=sys.stderr)
        for label in failed:
            print(f"  {label}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(EXPERIMENTS)} experiments completed.")


if __name__ == "__main__":
    main()
