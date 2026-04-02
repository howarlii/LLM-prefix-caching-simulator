# KV cache hit-rate simulator

Python simulator for **prefix KV cache** behavior on LLM serving workloads: page-granularity matching, radix-tree cache, and configurable eviction (LRU / LFU / FIFO). Built to mirror the experiment axes discussed in the Strata paper (arXiv:2508.18572): page size, cache capacity, request ordering, and eviction policy.

## Layout

- `src/radix_tree.py` — page-level radix tree (one node per page of token ids).
- `src/cache_simulator.py` — request loop, capacity enforcement, `MultiTierCacheSimulator` stub for future multi-tier host-cache modeling.
- `src/strategies/` — eviction policies implementing `EvictionStrategy`.
- `src/datasets_loader.py` — Hugging Face datasets → text requests (LooGLE, NarrativeQA, ShareGPT; ReviewMT skipped by default).
- `src/request_generator.py` — Qwen3 tokenizer, on-disk tokenization cache, ordering modes.
- `src/metrics.py` — aggregate metrics (page/token hit rate, per-request distribution, load/compute ratio, tree stats).
- `experiments/` — CLI sweeps and `run_all.py`.
- `analysis/plot_results.ipynb` — matplotlib plots from `results/*.csv`.

## Environment and caches

- **CPU-only**: dependencies target the **CPU build of PyTorch** (`--extra-index-url` in `requirements.txt`). `ensure_hf_cache_dirs()` also calls `ensure_cpu_only()` (`CUDA_VISIBLE_DEVICES=""`, `TOKENIZERS_PARALLELISM=false`) unless you set `KV_SIM_ALLOW_CUDA=1`.
- **Hugging Face / datasets cache**: `src/config.ensure_hf_cache_dirs()` sets `HF_HOME` and `HF_DATASETS_CACHE` under `/data/howarli/.cache/huggingface` when that path exists, so downloads do not fill `$HOME` (per design notes).
- **Tokenizer**: default `Qwen/Qwen3-0.6B` (override with env `KV_SIM_TOKENIZER` or CLI `--tokenizer`).
- **Tokenization cache**: `data/tokenized/*.jsonl` (gitignored under `data/*/`).

## Capacity in “GB”

Finite capacities are converted to a token budget using a configurable KV footprint (default `2 * 28 * 4096 * 2` bytes per token, fp16-style heuristic). Override bytes/token via experiment flags where exposed.

## Quick start

```bash
cd /path/to/llm_prefix_caching
python -m venv .venv && source .venv/bin/activate
# Prefer explicit CPU torch if pip still picks a CUDA wheel:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Page-size sweep (downloads data + tokenizer on first run)
python experiments/run_page_size.py --dataset loogle --page-sizes 32,64,128

# Ordering comparison
python experiments/run_ordering.py --dataset loogle --page-size 32

# Eviction vs capacity (GB)
python experiments/run_eviction.py --dataset loogle --capacities 20,40,80,160

# Small end-to-end check
python experiments/run_all.py --smoke --dataset loogle
```

Results go to `results/` as CSV + per-run JSON.

## Metrics (per configuration)

- Page-level and token-level cache hit rates.
- Per-request token hit rate: mean, p50, p90, p99.
- Load vs compute tokens and their ratio (when `compute_tokens > 0`).
- Peak / average cached token usage.
- Radix tree depth histogram and access-count summaries by depth.

## Orderings

- `original` — loader order.
- `min_distance` — group requests by `group_id` (tighter prefix locality).
- `max_distance` — round-robin across groups.
- `random` — shuffle.

## Optional / future

- **ReviewMT**: not wired (returns empty list); extend `datasets_loader.py` when a stable source is available.
- **Multi-tier cache**: `MultiTierCacheSimulator` raises `NotImplementedError` until implemented.
