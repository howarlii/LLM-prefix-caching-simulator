"""Download and normalize datasets into plain-text request payloads."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from src.config import ensure_hf_cache_dirs

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Mooncake traces: pre-baked block hash sequences (no tokenizer). See technical report §4.
_MOONCAKE_TRACE_JSONL: Dict[str, Path] = {
    "mooncake_toolagent": _REPO_ROOT / "data" / "mooncake_trace" / "toolagent_trace.jsonl",
    "mooncake_conversation": _REPO_ROOT / "data" / "mooncake_trace" / "conversation_trace.jsonl",
}
_MOONCAKE_TRACE_ALIASES: Dict[str, str] = {
    "toolagent_trace": "mooncake_toolagent",
    "conversation_trace": "mooncake_conversation",
}

# philschmid/sharegpt-raw: raw ShareGPT JSON (duplicated from jeffwan/sharegpt_vicuna).
_SHAREGPT_RAW_HF_REPO = "philschmid/sharegpt-raw"
_SHAREGPT_90K_JSON_FILES = (
    "sharegpt_90k_raw_dataset/sg_90k_part1.json",
    "sharegpt_90k_raw_dataset/sg_90k_part2.json",
)
_SHAREGPT_90K_RAW_ALIASES: Dict[str, str] = {
    "sharegpt_raw": "sharegpt_90k_raw",
    "philschmid_sharegpt_raw": "sharegpt_90k_raw",
}


def sharegpt_90k_raw_canonical_name(name: str) -> Optional[str]:
    """Return ``sharegpt_90k_raw`` if ``name`` refers to that corpus, else ``None``."""
    n = name.lower()
    if n == "sharegpt_90k_raw":
        return "sharegpt_90k_raw"
    return _SHAREGPT_90K_RAW_ALIASES.get(n)


def mooncake_trace_canonical_name(name: str) -> Optional[str]:
    """Return canonical dataset key if ``name`` is a Mooncake JSONL trace, else ``None``."""
    n = name.lower()
    if n in _MOONCAKE_TRACE_JSONL:
        return n
    mapped = _MOONCAKE_TRACE_ALIASES.get(n)
    return mapped


def is_mooncake_trace_dataset(name: str) -> bool:
    return mooncake_trace_canonical_name(name) is not None


def mooncake_trace_jsonl_path(name: str) -> Path:
    """Resolved path to the trace file for a Mooncake dataset name or alias."""
    c = mooncake_trace_canonical_name(name)
    if c is None:
        raise ValueError(f"Not a Mooncake trace dataset: {name!r}")
    return _MOONCAKE_TRACE_JSONL[c]


def _stable_text_digest(text: str, n_hex: int = 16) -> str:
    """Stable fingerprint for cache keys (do not use built-in ``hash()`` — it is salted per process)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n_hex]


@dataclass
class RawRequest:
    """One logical request: concatenated input text and a stable group key."""

    text: str
    group_id: str
    meta: Dict[str, Any]


def _iter_loogle() -> Iterator[RawRequest]:
    ensure_hf_cache_dirs()
    from datasets import load_dataset

    ds = load_dataset("bigai-nlco/LooGLE", "shortdep_qa", split="test")
    for i, row in enumerate(ds):
        ctx = row.get("context") or ""
        q = row.get("question") or ""
        title = row.get("title") or ""
        gid = f"loogle:{title}:{_stable_text_digest(ctx)}"
        text = f"{ctx}{q}"
        yield RawRequest(text=text, group_id=gid, meta={"dataset": "loogle", "idx": i})


def _narrativeqa_doc_text(row: dict) -> str:
    doc = row.get("document")
    if isinstance(doc, str):
        return doc.strip()
    if isinstance(doc, dict):
        return (doc.get("text") or "").strip()
    return ""


def _count_tokens_approx(text: str) -> int:
    """Rough length check before heavy tokenization (tiktoken cl100k)."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _iter_narrativeqa(
    max_tokens: int = 128_000,
    num_documents: int = 50,
    seed: int = 0,
) -> Iterator[RawRequest]:
    ensure_hf_cache_dirs()
    from datasets import load_dataset

    ds = load_dataset("deepmind/narrativeqa", split="test")
    rng = random.Random(seed)
    # Group by document id
    by_doc: Dict[str, List[dict]] = {}
    for row in ds:
        doc = row.get("document") or {}
        doc_id = str(doc.get("id", ""))
        if not doc_id:
            continue
        by_doc.setdefault(doc_id, []).append(row)

    doc_ids = [d for d, rows in by_doc.items() if rows]
    rng.shuffle(doc_ids)

    picked: List[str] = []
    for did in doc_ids:
        if len(picked) >= num_documents:
            break
        sample_row = by_doc[did][0]
        text_body = _narrativeqa_doc_text(sample_row)
        if not text_body:
            continue
        if _count_tokens_approx(text_body) > max_tokens:
            continue
        picked.append(did)

    for did in picked:
        for j, row in enumerate(by_doc[did]):
            text_body = _narrativeqa_doc_text(row)
            qobj = row.get("question") or {}
            if isinstance(qobj, str):
                qtext = qobj.strip()
            else:
                qtext = (qobj.get("text") or "").strip()
            text = f"{text_body}{qtext}"
            yield RawRequest(
                text=text,
                group_id=f"narrativeqa:{did}",
                meta={"dataset": "narrativeqa", "doc": did, "q": j},
            )


def _normalize_sharegpt_conversation_list(conv: Any) -> List[Any]:
    """Coerce HF / JSON conversation rows into a list of turn dicts (with optional ``value``)."""
    if conv is None:
        return []
    if hasattr(conv, "tolist"):
        conv = conv.tolist()
    if not isinstance(conv, list) or not conv:
        return []
    out: List[Any] = []
    for turn in conv:
        if isinstance(turn, str):
            try:
                turn = json.loads(turn)
            except json.JSONDecodeError:
                continue
        if isinstance(turn, dict):
            out.append(turn)
    return out


def _yield_sharegpt_style_requests(
    conv_idx: int,
    conv: Any,
    *,
    meta_dataset: str,
    group_prefix: str,
) -> Iterator[RawRequest]:
    turns = _normalize_sharegpt_conversation_list(conv)
    if not turns:
        return
    acc: List[str] = []
    for t, turn in enumerate(turns):
        val = turn.get("value") or ""
        acc.append(val)
        text = "".join(acc)
        yield RawRequest(
            text=text,
            group_id=f"{group_prefix}:{conv_idx}",
            meta={"dataset": meta_dataset, "conv": conv_idx, "turn": t},
        )


def _iter_sharegpt(max_conversations: int = 10_000, seed: int = 0) -> Iterator[RawRequest]:
    ensure_hf_cache_dirs()
    from datasets import load_dataset

    # Streaming avoids loading the full corpus into RAM.
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split="train",
        streaming=True,
    )
    convo_count = 0
    for idx, row in enumerate(ds):
        if convo_count >= max_conversations:
            break
        conv = row.get("conversations")
        turns = _normalize_sharegpt_conversation_list(conv)
        if not turns:
            continue
        convo_count += 1
        yield from _yield_sharegpt_style_requests(
            idx, turns, meta_dataset="sharegpt", group_prefix="sharegpt"
        )


def _iter_sharegpt_90k_raw(
    max_conversations: int = 10_000,
    seed: int = 0,
) -> Iterator[RawRequest]:
    """Stream philschmid/sharegpt-raw 90k JSON (sharegpt_90k_raw_dataset/, two parts) from Hugging Face."""
    del seed  # reserved for API parity with other loaders
    ensure_hf_cache_dirs()
    import ijson
    from huggingface_hub import hf_hub_download

    conv_idx = 0
    convo_count = 0
    for rel in _SHAREGPT_90K_JSON_FILES:
        path = hf_hub_download(
            repo_id=_SHAREGPT_RAW_HF_REPO,
            filename=rel,
            repo_type="dataset",
        )
        with open(path, "rb") as f:
            for row in ijson.items(f, "item"):
                if convo_count >= max_conversations:
                    return
                if not isinstance(row, dict):
                    continue
                conv = row.get("conversations")
                turns = _normalize_sharegpt_conversation_list(conv)
                if not turns:
                    continue
                convo_count += 1
                yield from _yield_sharegpt_style_requests(
                    conv_idx,
                    turns,
                    meta_dataset="sharegpt_90k_raw",
                    group_prefix="sharegpt_90k_raw",
                )
                conv_idx += 1


def _yield_chat_messages_requests(
    conv_idx: int,
    messages: Any,
    *,
    meta_dataset: str,
    group_prefix: str,
) -> Iterator[RawRequest]:
    """Yield cumulative-prefix requests from a ``[{role, content}, ...]`` message list."""
    if not isinstance(messages, list) or not messages:
        return
    acc: List[str] = []
    for t, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content") or ""
        acc.append(content)
        text = "".join(acc)
        yield RawRequest(
            text=text,
            group_id=f"{group_prefix}:{conv_idx}",
            meta={"dataset": meta_dataset, "conv": conv_idx, "turn": t},
        )


def _iter_swe_smith(max_conversations: int = 10_000) -> Iterator[RawRequest]:
    """Stream SWE-smith 66k trajectories (role/content chat format)."""
    ensure_hf_cache_dirs()
    from datasets import load_dataset

    ds = load_dataset(
        "Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k",
        split="train",
        streaming=True,
    )
    convo_count = 0
    for idx, row in enumerate(ds):
        if convo_count >= max_conversations:
            break
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        convo_count += 1
        yield from _yield_chat_messages_requests(
            idx, messages, meta_dataset="swe_smith", group_prefix="swe_smith"
        )


def _iter_mooncake_trace(jsonl_path: Path, dataset_key: str) -> Iterator[RawRequest]:
    """Load Mooncake-style traces: each line has timestamp, lengths, and hash_ids (block ids)."""
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Mooncake trace not found: {jsonl_path}")

    raw_rows: List[tuple[int, int, Dict[str, Any]]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Expected JSON per line in {jsonl_path} (line {line_idx + 1}): {e}"
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object per line in {jsonl_path}, line {line_idx + 1}")
            raw_rows.append((line_idx, int(obj.get("timestamp", 0)), obj))

    raw_rows.sort(key=lambda x: (x[1], x[0]))

    for out_idx, (_, _ts, obj) in enumerate(raw_rows):
        h = obj.get("hash_ids")
        if not isinstance(h, list) or not h:
            continue
        try:
            block_ids = [int(x) for x in h]
        except (TypeError, ValueError):
            continue
        meta: Dict[str, Any] = {
            "dataset": dataset_key,
            "trace_idx": out_idx,
            "timestamp": obj.get("timestamp"),
            "input_length": obj.get("input_length"),
            "output_length": obj.get("output_length"),
            "hash_ids": block_ids,
        }
        yield RawRequest(
            text="",
            group_id=f"{dataset_key}:{out_idx}",
            meta=meta,
        )


def load_raw_requests(
    name: str,
    *,
    narrativeqa_docs: int = 50,
    sharegpt_conversations: int = 10_000,
    seed: int = 0,
) -> List[RawRequest]:
    """Materialize a dataset into a list of :class:`RawRequest`."""
    name = name.lower()
    if name == "reviewmt":
        return []
    if name == "loogle":
        return list(_iter_loogle())
    if name == "narrativeqa":
        return list(_iter_narrativeqa(num_documents=narrativeqa_docs, seed=seed))
    if name == "sharegpt":
        return list(_iter_sharegpt(max_conversations=sharegpt_conversations, seed=seed))
    if name in ("swe_smith", "swesmith", "swe-smith"):
        return list(
            _iter_swe_smith(max_conversations=sharegpt_conversations)
        )
    if sharegpt_90k_raw_canonical_name(name) is not None:
        return list(
            _iter_sharegpt_90k_raw(
                max_conversations=sharegpt_conversations,
                seed=seed,
            )
        )
    moon_key = mooncake_trace_canonical_name(name)
    if moon_key is not None:
        return list(_iter_mooncake_trace(mooncake_trace_jsonl_path(name), moon_key))
    raise ValueError(
        f"Unknown dataset {name!r}; choose from loogle, narrativeqa, sharegpt, sharegpt_90k_raw, "
        f"swe_smith, reviewmt, mooncake_toolagent, mooncake_conversation (aliases: toolagent_trace, "
        f"conversation_trace, sharegpt_raw, philschmid_sharegpt_raw, swesmith, swe-smith)"
    )


def save_manifest(path: Path, requests: Sequence[RawRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"text": r.text, "group_id": r.group_id, "meta": r.meta} for r in requests
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_manifest(path: Path) -> List[RawRequest]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        RawRequest(text=x["text"], group_id=x["group_id"], meta=x.get("meta", {}))
        for x in data
    ]
