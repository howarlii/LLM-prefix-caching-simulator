#!/usr/bin/env python3
"""Run bundled experiment scripts sequentially (optional smoke mode)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Run page_size, ordering, and eviction sweeps")
    p.add_argument("--smoke", action="store_true", help="Small sweeps for a quick sanity check")
    p.add_argument("--dataset", default="loogle")
    args = p.parse_args()

    py = sys.executable
    base = [py, "-u", str(_ROOT / "experiments" / "run_page_size.py"), "--dataset", args.dataset]
    if args.smoke:
        base += ["--page-sizes", "32,64", "--max-requests", "200"]
    _run(base)

    ord_cmd = [py, "-u", str(_ROOT / "experiments" / "run_ordering.py"), "--dataset", args.dataset]
    if args.smoke:
        ord_cmd += ["--orderings", "original,min_distance", "--max-requests", "200"]
    _run(ord_cmd)

    ev_cmd = [py, "-u", str(_ROOT / "experiments" / "run_eviction.py"), "--dataset", args.dataset]
    if args.smoke:
        ev_cmd += ["--capacities", "20,40", "--strategies", "lru,fifo", "--max-requests", "200"]
    _run(ev_cmd)


if __name__ == "__main__":
    main()
