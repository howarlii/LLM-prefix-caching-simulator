"""Tokenize requests, cache token ids, and apply ordering policies."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

from tqdm import tqdm

from src.config import DEFAULT_TOKENIZER_NAME, TOKEN_CACHE_DIR, ensure_cpu_only
from src.datasets_loader import RawRequest

OrderingName = Literal["original", "min_distance", "max_distance", "random"]


@dataclass
class TokenizedRequest:
    """Request with materialized token ids and grouping metadata."""

    token_ids: List[int]
    group_id: str
    meta: dict


# ── Module-level caches ──────────────────────────────────────────────────────
_tokenizer_cache: Dict[str, object] = {}
_MODEL_MAX_LENGTH_CACHE_FILE = TOKEN_CACHE_DIR / ".model_max_lengths.json"


def _load_tokenizer(name: str):
    """Load tokenizer with in-process caching (avoids repeated from_pretrained)."""
    if name in _tokenizer_cache:
        return _tokenizer_cache[name]
    ensure_cpu_only()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    _tokenizer_cache[name] = tok
    return tok


def _capped_model_max_length(tok) -> int:
    m = getattr(tok, "model_max_length", None)
    if m is None or m > 1_000_000:
        m = 131_072
    return m


def _get_encode_max_length(tokenizer_name: str) -> int:
    """Compute encode_max_length, avoiding full tokenizer load via disk cache.

    On the first call for a given tokenizer the model is loaded and the
    ``model_max_length`` is persisted to ``TOKEN_CACHE_DIR/.model_max_lengths.json``.
    Subsequent calls (even across processes) read the disk cache and skip the
    expensive ``AutoTokenizer.from_pretrained``.
    """
    m: int | None = None
    # 1. Try disk cache
    try:
        if _MODEL_MAX_LENGTH_CACHE_FILE.is_file():
            data = json.loads(_MODEL_MAX_LENGTH_CACHE_FILE.read_text(encoding="utf-8"))
            if tokenizer_name in data:
                m = int(data[tokenizer_name])
    except Exception:
        pass

    # 2. Fall back to loading the tokenizer (and persist the result)
    if m is None:
        tok = _load_tokenizer(tokenizer_name)
        m = _capped_model_max_length(tok)
        try:
            disk: dict = {}
            if _MODEL_MAX_LENGTH_CACHE_FILE.is_file():
                disk = json.loads(_MODEL_MAX_LENGTH_CACHE_FILE.read_text(encoding="utf-8"))
            disk[tokenizer_name] = m
            _MODEL_MAX_LENGTH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _MODEL_MAX_LENGTH_CACHE_FILE.write_text(
                json.dumps(disk, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    # 3. Apply env-var override
    raw = os.environ.get("KV_SIM_MAX_INPUT_TOKENS", "").strip()
    if raw.isdigit():
        u = int(raw)
        if u > 0:
            return min(u, m)
    return m


def _encode_max_length(tok) -> int:
    """Per-sequence cap for tokenization (min of env, tokenizer model_max_length)."""
    m = getattr(tok, "model_max_length", None)
    if m is None or m > 1_000_000:
        m = 131_072
    raw = os.environ.get("KV_SIM_MAX_INPUT_TOKENS", "").strip()
    if raw.isdigit():
        u = int(raw)
        if u > 0:
            return min(u, m)
    return m


def _chunked(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


# One tokenizer per spawn worker — avoid AutoTokenizer.from_pretrained on every batch
# (Qwen3 path can call the Hub model_info API; 10k+ loads → 429 rate limit).
_worker_tok = None
_worker_encode_max_len: int = 131_072


def _tokenize_worker_init(args: Tuple[str, int]) -> None:
    tokenizer_name, encode_max_len = args
    global _worker_tok, _worker_encode_max_len
    _worker_tok = _load_tokenizer(tokenizer_name)
    _worker_encode_max_len = encode_max_len


def _tokenize_worker(texts: List[str]) -> List[List[int]]:
    assert _worker_tok is not None
    out = _worker_tok(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=_worker_encode_max_len,
    )
    return out["input_ids"]


def tokenize_texts_parallel(
    texts: List[str],
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    batch_size: int = 16,
    num_workers: int = 0,
    *,
    encode_max_length: int,
) -> List[List[int]]:
    """Tokenize many strings; uses processes when ``num_workers > 1``."""
    if not texts:
        return []
    if num_workers <= 0:
        num_workers = max(1, (os.cpu_count() or 4))

    if num_workers == 1:
        tok = _load_tokenizer(tokenizer_name)
        result: List[List[int]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", unit="batch"):
            chunk = texts[i : i + batch_size]
            batch = tok(
                chunk,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=encode_max_length,
            )
            result.extend(batch["input_ids"])
        return result

    chunks = _chunked(texts, batch_size)
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=num_workers,
        initializer=_tokenize_worker_init,
        initargs=((tokenizer_name, encode_max_length),),
    ) as pool:
        nested = list(
            tqdm(
                pool.imap(_tokenize_worker, chunks),
                total=len(chunks),
                desc="Tokenizing (parallel)",
                unit="batch",
            )
        )
    flat: List[List[int]] = []
    for part in nested:
        flat.extend(part)
    return flat


def _cache_key_parts(
    dataset_name: str,
    tokenizer_name: str,
    requests: Sequence[RawRequest],
    *,
    encode_max_length: int,
) -> str:
    """Return a hash that is stable for any prefix-subset of the same base dataset.

    ``len(requests)`` is intentionally excluded from the hash so that runs with
    different ``--max-requests`` values share the same base key.  The count is
    embedded separately in the cache filename instead.
    """
    h = hashlib.sha256()
    h.update(dataset_name.encode())
    h.update(b"\0")
    h.update(tokenizer_name.encode())
    h.update(b"\0")
    h.update(str(encode_max_length).encode())
    h.update(b"\0")
    n = len(requests)
    if n == 0:
        return h.hexdigest()[:24]
    # Use the first min(n, 200) requests as a prefix-stable fingerprint.
    # This is invariant across different --max-requests values drawn from the
    # same base dataset ordering.
    sample_end = min(n, 200)
    for i in range(sample_end):
        r = requests[i]
        h.update(r.group_id.encode())
        h.update(str(len(r.text)).encode())
        h.update(b"\0")
    return h.hexdigest()[:24]


def load_or_tokenize(
    dataset_name: str,
    requests: List[RawRequest],
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    *,
    num_workers: int = 0,
    batch_size: int = 16,
    force_recompute: bool = False,
) -> List[TokenizedRequest]:
    """Load token ids from disk cache or compute and save."""
    TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Fast path: resolve encode_max_length from disk cache (skips tokenizer load).
    encode_max_length = _get_encode_max_length(tokenizer_name)
    key = _cache_key_parts(
        dataset_name, tokenizer_name, requests, encode_max_length=encode_max_length
    )
    n = len(requests)
    cache_path = TOKEN_CACHE_DIR / f"{dataset_name}_{key}_n{n}.jsonl"

    def _try_load(path: Path, want: int) -> List[TokenizedRequest]:
        """Read the first ``want`` lines from a cache file; return [] on mismatch."""
        out: List[TokenizedRequest] = []
        show_bar = want >= 1000 and sys.stderr.isatty()
        with path.open(encoding="utf-8") as f:
            it = zip(requests, f)
            if show_bar:
                it = tqdm(it, total=want, desc="Loading cached tokens", leave=False, unit="req")
            for req, line in it:
                row = json.loads(line)
                out.append(
                    TokenizedRequest(
                        token_ids=row["token_ids"],
                        group_id=req.group_id,
                        meta={**req.meta, **row.get("extra_meta", {})},
                    )
                )
        return out if len(out) == want else []

    if not force_recompute:
        # 1. Exact match.
        if cache_path.is_file():
            out = _try_load(cache_path, n)
            if out:
                return out

        # 2. Superset: any cached file for the same dataset+key with count >= n.
        prefix = f"{dataset_name}_{key}_n"
        for candidate in sorted(TOKEN_CACHE_DIR.glob(f"{prefix}*.jsonl")):
            stem = candidate.stem  # e.g. "loogle_abc123_n2000"
            try:
                cached_n = int(stem[len(prefix):])
            except ValueError:
                continue
            if cached_n >= n:
                out = _try_load(candidate, n)
                if out:
                    return out

    texts = [r.text for r in requests]
    ids_list = tokenize_texts_parallel(
        texts,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        num_workers=num_workers,
        encode_max_length=encode_max_length,
    )
    if len(ids_list) != len(requests):
        raise RuntimeError("tokenization length mismatch")

    with cache_path.open("w", encoding="utf-8") as f:
        for req, tids in zip(requests, ids_list):
            f.write(
                json.dumps(
                    {"token_ids": tids, "extra_meta": {}},
                    ensure_ascii=False,
                )
                + "\n"
            )

    return [
        TokenizedRequest(token_ids=tids, group_id=r.group_id, meta=dict(r.meta))
        for r, tids in zip(requests, ids_list)
    ]


def order_requests(
    items: List[TokenizedRequest],
    mode: OrderingName,
    seed: int = 0,
) -> List[TokenizedRequest]:
    """Reorder tokenized requests for cache-distance experiments."""
    if mode == "original":
        return list(items)

    rng = random.Random(seed)
    by_g: Dict[str, List[TokenizedRequest]] = defaultdict(list)
    for x in items:
        by_g[x.group_id].append(x)

    groups = list(by_g.keys())
    if mode == "random":
        # Shuffle inter-group ordering while preserving intra-group turn order.
        # This models random conversation arrival times but causal turn dependencies.
        queues = {g: deque(by_g[g]) for g in groups}
        active = list(groups)
        rng.shuffle(active)
        out: List[TokenizedRequest] = []
        while active:
            # Pick a random active group and take its next turn.
            idx = rng.randrange(len(active))
            g = active[idx]
            out.append(queues[g].popleft())
            if not queues[g]:
                active.pop(idx)
        return out

    if mode == "min_distance":
        # All requests of a group appear consecutively (stable within group).
        out = []
        for g in sorted(groups):
            out.extend(by_g[g])
        return out

    if mode == "max_distance":
        # Round-robin across groups so same-group requests are spread out.
        deques = {g: deque(by_g[g]) for g in groups}
        order = list(groups)
        rng.shuffle(order)
        out = []
        while True:
            moved = False
            for g in order:
                if deques[g]:
                    out.append(deques[g].popleft())
                    moved = True
            if not moved:
                break
        return out

    raise ValueError(f"Unknown ordering {mode!r}")
