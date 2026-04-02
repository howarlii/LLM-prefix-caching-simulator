"""Shared configuration: HF cache dirs, tokenizer id, KV capacity heuristics."""

from __future__ import annotations

import os
from pathlib import Path

# Prefer /data/howarli for HuggingFace and dataset caches (avoid filling $HOME).
_DEFAULT_DATA_ROOT = Path("/data/howarli")
_HOME = Path.home()


def ensure_cpu_only() -> None:
    """Use CPU only: hide GPUs from this process (tokenizer / optional torch).

    Set ``KV_SIM_ALLOW_CUDA=1`` to skip this (not used in this project by default).
    """
    if os.environ.get("KV_SIM_ALLOW_CUDA", "").lower() in ("1", "true", "yes"):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Avoid tokenizer C++ side oversubscribing CPU threads when using multiprocessing.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_hf_cache_dirs() -> None:
    """Point HF and datasets caches under ``/data/howarli`` when home is default."""
    ensure_cpu_only()
    data_root = _DEFAULT_DATA_ROOT if _DEFAULT_DATA_ROOT.is_dir() else _HOME
    hf_home = data_root / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    # Transformers uses HF_HOME for hub downloads


# Approximate KV bytes per token (fp16), ~28 layers × 4096 dim × 2 (K+V) × 2 bytes
KV_BYTES_PER_TOKEN_DEFAULT = 2 * 28 * 4096 * 2


def gb_to_token_capacity(gb: float, kv_bytes_per_token: int = KV_BYTES_PER_TOKEN_DEFAULT) -> int:
    """Convert cache capacity in GB to an equivalent token budget."""
    if gb <= 0 or gb == float("inf"):
        raise ValueError("gb must be positive and finite")
    return int(gb * (1024**3) / kv_bytes_per_token)


# Qwen3 family tokenizer (design doc: qwen-3.5; use small checkpoint for tokenizer-only).
DEFAULT_TOKENIZER_NAME = os.environ.get("KV_SIM_TOKENIZER", "Qwen/Qwen3-0.6B")

# Hard cap on encoded input length (tail truncated; head kept — matches default truncation_side).
# If unset, use each tokenizer's model_max_length (capped sanely). Override example: KV_SIM_MAX_INPUT_TOKENS=65536

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TOKEN_CACHE_DIR = DATA_DIR / "tokenized"
RESULTS_DIR = PROJECT_ROOT / "results"
