# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python simulator for **prefix KV cache behavior** on LLM serving workloads. It mirrors experiment axes from the Strata paper (arXiv:2508.18572): page size, cache capacity, request ordering, and eviction policy. The core question: given a stream of tokenized LLM requests, what fraction of prefix tokens can be served from a shared radix-tree cache?

## Environment Setup

```bash
cd /data/howarli/dev/llm_prefix_caching
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **CPU-only by default**: `ensure_cpu_only()` sets `CUDA_VISIBLE_DEVICES=""` and `TOKENIZERS_PARALLELISM=false`. Override with `KV_SIM_ALLOW_CUDA=1`.
- **HF cache** redirected to `/data/howarli/.cache/huggingface` (set by `ensure_hf_cache_dirs()`). HuggingFace tokenizers and datasets are downloaded on first run.
- **Tokenization cache**: results are stored in `data/tokenized/*.jsonl` (gitignored). Delete to force re-tokenization.

## Common Commands

```bash
# Page size sweep
python experiments/run_page_size.py --dataset loogle --page-sizes 32,64,128

# Request ordering experiment
python experiments/run_ordering.py --dataset loogle --page-size 32

# Eviction policy vs capacity
python experiments/run_eviction.py --dataset loogle --capacities 20,40,80,160

# Smoke test (small run)
python experiments/run_all.py --smoke --dataset loogle
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KV_SIM_TOKENIZER` | `Qwen/Qwen3-0.6B` | HuggingFace tokenizer name |
| `KV_SIM_MAX_INPUT_TOKENS` | tokenizer's `model_max_length` | Truncation length for tokenization |
| `KV_SIM_ALLOW_CUDA` | `""` (CPU-only) | Set to `1` to allow GPU access |

### Output

Results are written to `results/` as CSV summaries + per-run JSON files:
- CSV: `results/page_size_sweep_<dataset>.csv`, etc.
- JSON: `results/page_size_json_<dataset>/<slug>.json`

## Architecture

### Data Flow (one simulation run)

```
RawRequest (text + group_id)
  → request_generator.load_or_tokenize()  [cached in data/tokenized/]
  → TokenizedRequest (token_ids + group_id)
  → runner.order_requests()  [reorder by mode]
  → KVCacheSimulator.process_token_ids()
      → RadixTree.simulate_request()  [prefix match + insert]
      → EvictionStrategy.select_nodes()  [when over capacity]
  → metrics.compute_run_metrics()
```

### Key Classes

**`RadixTree` (`src/radix_tree.py`)**
- Each node = one page (tuple of token ids). Root node holds an empty page.
- `simulate_request(pages)`: walks the tree as far as the prefix matches (cache hit), then inserts the remaining suffix pages (cache miss + fill). Touches hit nodes to update `last_access` and `access_count`. Marks the last node as `is_turn_end` for multi-turn continuity tracking.
- `remove_leaf(node)`: removes a leaf and recursively prunes its ancestors up to the nearest branching point or root (coalesces single-child chains).
- `leaf_nodes()`: returns all evictable leaves. The empty root is never evicted.
- `valid_cached_depth_histogram()`: key metric for branching mass — nodes with >1 child contribute (x-1) to their depth, representing wasted/shared cache capacity.

**`KVCacheSimulator` (`src/cache_simulator.py`)**
- Owns a primary HBM `RadixTree` and an `EvictionStrategy`. An optional DRAM tier (second `RadixTree` + its own strategy) is enabled when `dram_strategy is not None` AND `dram_capacity_tokens > 0`; otherwise it runs as a single-tier HBM cache.
- `process_token_ids(token_ids)`: splits into pages → `tree.simulate_request()` (HBM) → optional read-only `dram_tree.prefix_match()` for extra hits → `_evict_until_fit()` (HBM eviction demotes leaf paths to DRAM, then DRAM eviction discards).

**`EvictionStrategy` (`src/strategies/base.py`)**
- Abstract interface: `select_nodes(tree, num_nodes) → List[RadixNode]`.
- Implementations: `LRUStrategy` (sorts leaves by `last_access`), `LFUStrategy` (sorts by `access_count`), `FIFOStrategy` (sorts by `creation_order`).
- To add a new eviction policy: implement `EvictionStrategy` in `src/strategies/` and register it in `experiments/runner.strategy_from_name()`.

**`request_generator.py`**
- `load_or_tokenize()`: disk-cached tokenization using Qwen3 tokenizer. Cache key is a SHA-256 hash of (dataset, tokenizer, max_length, request count, group_id, text length, first/last entries). Parallel tokenization via `mp.get_context("spawn")` when `num_workers > 1`.
- `order_requests()`: `min_distance` groups same-group requests consecutively; `max_distance` round-robins across groups; `random` shuffles; `original` preserves loader order.

**`datasets_loader.py`**
- Supported datasets: `loogle`, `narrativeqa`, `sharegpt`, `sharegpt_90k_raw`, `mooncake_toolagent`, `mooncake_conversation`. `reviewmt` returns empty (not wired).
- Mooncake traces bypass the tokenizer — `hash_ids` in the JSONL are used directly as token ids. For these datasets, `effective_page_size()` forces `page_size=1`.
- `load_raw_requests()` is the entry point; results can be saved/loaded via `save_manifest()` / `load_manifest()`.

**`metrics.py`**
- `compute_run_metrics()` aggregates: per-tier hit rates (`hbm_token_hit_rate`, `dram_token_hit_rate`), load/compute ratio, peak/avg cached tokens (HBM and DRAM), tree depth histogram, access-count percentiles per depth, and wall-clock savings.
- **Wall-clock model**: GPU compute time = `flop / gpu_flops`. DRAM cache hits incur a PCIe transfer cost = `dram_hit_bytes / pcie_bandwidth`. `total_saved_time = gpu_compute_time_no_cache − gpu_compute_time_with_cache − pcie_total_transfer_time`. Per-request percentiles are reported as `per_request_saved_time_{mean,p50,p90,p99}`.
- **Hardware throughput** (`gpu_flops`, `pcie_bandwidth`) are passed as keyword arguments to `compute_run_metrics()` and `run_simulation()` — they are *not* fields on `ModelConfig`. Defaults live in `src/config.py` (`DEFAULT_GPU_FLOPS`, `DEFAULT_PCIE_BANDWIDTH`); CLI overrides via `run_one.py`'s `--gpu-flops` / `--pcie-bandwidth` flags or `run_all.py`'s `GPU_FLOPS` / `PCIE_BANDWIDTH` constants.
- **Naming convention**: `flop` = a count of floating-point ops (extensive); `flops` = ops/sec (rate). Use `flop_save_rate`, `total_flop_*`, `prefill_flop()`, but `gpu_flops` (a throughput).

## KV Capacity Heuristic

Capacity in GB is converted to token budget using `2 * 28 * 4096 * 2` bytes/token (fp16, 28-layer 4096-dim model). Override with `--kv-bytes-per-token` on experiment scripts.

## Adding a New Eviction Strategy

1. Create `src/strategies/my_strategy.py` implementing `select_nodes(self, tree: RadixTree, num_nodes: int) → List[RadixNode]`.
2. Add `from src.strategies.my_strategy import MyStrategy` and a branch in `experiments/runner.strategy_from_name()`.
3. Use it with `--strategy my_strategy` in any experiment script.
