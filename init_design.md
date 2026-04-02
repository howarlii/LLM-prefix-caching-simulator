# Prompt for Claude Code: KV Cache Hit Rate Simulator

## 项目目标

构建一个 Python 项目，用于模拟 LLM Serving 中不同 KV Cache 策略下的 **prefix cache hit rate**。使用 Strata（arXiv:2508.18572）论文中使用的数据集，在不同参数配置下测量 cache hit rate，帮助理解 page size、cache capacity、eviction policy、request ordering 等因素对 cache 效率的影响。

## 核心概念

### Prefix Cache Matching 规则
- KV Cache 以 **page** 为单位进行管理，每个 page 包含固定数量的 token（即 `page_size`）
- 两个 request 的 prefix matching 是 **逐 page 比较** 的：只有一个 page 内的所有 token 都完全一致，这个 page 才算 cache hit
- 例如：两个 request 共享 1000 个 token 的前缀，page_size=32 时，前 31 个 page（992 token）全部命中，第 32 个 page（token 993-1024）如果有任何 token 不同则整个 page miss
- Cache hit 的 token 不需要重新计算 prefill，只需要从缓存加载；cache miss 的 token 需要重新计算

### 指标定义
- **Token-level cache hit rate** = 命中的 cached token 总数 / 所有 request 的总 input token 数
- **Page-level cache hit rate** = 命中的 cached page 数 / 所有 request 需要的总 page 数
- **Per-request cache hit rate** = 每个 request 的 hit token / 该 request 的 total input token

## 数据集

使用以下 4 个数据集，通过 HuggingFace `datasets` 库下载：
下载之前先确认一下，默认的缓存地址是不是在/home/howarli/底下，如果是的话麻烦改到/data/howarli/底下，避免占用 home 目录的空间。

### 1. LooGLE (长文档 QA)
```python
from datasets import load_dataset
# 使用 shortdep_qa 子集（Wikipedia portion）
data = load_dataset('bigai-nlco/LooGLE', 'shortdep_qa', split='test')
```
- 结构：每条数据有 `context`（长文档）、`title`、`question`、`answer`
- **构造 request 的方式**：每个 request 的 input = context + question。同一个 title/context 下会有多个不同 question，这些 request 共享相同的 context 前缀
- 参考论文统计：105 个 context，2410 个 query，avg input ~21613 token

### 2. NarrativeQA (超长文档阅读理解)
```python
from datasets import load_dataset
data = load_dataset('deepmind/narrativeqa', split='test')
```
- 结构：每条数据有 `document.text`（完整故事/剧本）、`question.text`、`answers`
- **构造 request 的方式**：input = document.text + question.text。同一个 document 下有多个 question
- 注意：过滤掉超过 128K token 的文档（使用 tiktoken 或简单按空格/字符估算），采样 50 个文档
- 参考论文统计：50 个 context，1461 个 query，avg input ~54797 token

### 3. ReviewMT (多轮 agent 对话)
```python
# 从 GitHub 下载：https://github.com/chengtan9907/ReviewMT
# 或者如果有 HuggingFace 版本则使用 HF
```
- 结构：多轮对话，每轮 context 累积增长
- **构造 request 的方式**：每一轮对话是一个 request，input = 之前所有轮次的内容 + 当前轮的内容。同一个 session 内的连续轮次自然形成 prefix sharing
- 参考论文统计：100 个对话 session，1092 个 query，avg input ~17708 token
- 如果下载困难，可以先跳过这个数据集

### 4. ShareGPT (短上下文多轮对话)
```python
from datasets import load_dataset
data = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered')
```
- 结构：多轮对话，每个 conversation 有多个 turn
- **构造 request 的方式**：每个 turn 是一个 request，input = 之前所有 turn 的拼接 + 当前 turn。同 session 内 turn 之间有 prefix sharing
- 参考论文统计：200869 个 query，avg input ~681 token
- 注意：这个数据集很大，可以采样一个子集（比如前 10000 个 conversation）来测试

## Tokenization

使用 qwen-3.5 的tokenizer。
tokenize 之后得到 token id 序列，后续所有的 page 划分和 prefix matching 都基于 token id 序列。

## 模拟器核心逻辑

### Cache 数据结构
使用 **RadixTree**（前缀树）来模拟 prefix cache，这是 SGLang 使用的数据结构：
- 树的每个节点存储一个 page（`page_size` 个 token id），并记录hash值以便快速比较
- 查找时从根节点开始逐 page 匹配
- 匹配到的最长前缀就是 cache hit 的部分
- 每个节点中还需要记录一些元信息（比如最后访问时间、访问频率等）以支持不同的 eviction policy


### Cache 容量与 Eviction
- Cache 有容量限制（以 token 数或 page 数为单位）
- 当 cache 满时需要 evict，需要调用strategy中的接口，告知strategy需要evict几个节点，strategy根据自己的eviction policy决定evict哪些节点，并调用RadixTree提供的接口删除这些节点。
<!-- - Eviction 粒度：evict 整个 leaf-to-diverge-point 的路径（即 RadixTree 中不再被任何其他 sequence 共享的尾部 page） -->

### Strategy 接口设计
- 定义一个抽象基类 `EvictionStrategy`，所有 eviction policy（LRU, LFU, FIFO）都实现这个接口。


### 模拟流程
```
for each request in request_stream:
    1. tokenize request input
    2. 将 token 序列按 page_size 划分为 page 序列
    3. 在 RadixTree 中查找最长前缀匹配 → 得到 hit_pages 和 miss_pages
    4. 记录 hit/miss 统计
    5. 将完整的 token 序列插入 RadixTree（模拟 prefill 完成后 cache 被填充）
    6. 如果 cache 超出容量限制，执行 eviction
```

### 多层级缓存模拟（可选扩展）
模拟主机上两层缓存（例如快/慢两级 DRAM 或不同 NUMA / 内存池），不依赖 GPU：
- L1: 小容量，hit 时视为无额外 I/O 开销
- L2: 大容量，hit 时有 I/O 开销（可以简单记录需要从 L2 load 的 token 数, block 数，以及次数）
- eviction 从 L1 → L2，L2 满了再真正丢弃，L1 -> L2 的 也不一定是全量直接复制过去的，也有可能涉及 eviction policy 的选择（比如 LRU 可能会优先 evict 那些不常用的 page）
这一部分多层级缓存先不管，留个接口吧。

## 实验变量

### 必须测试的维度

1. **Page Size**: [1, 16, 32, 64, 128, 256, 512, 1024]
   - 核心变量，直接影响 cache hit rate 和 I/O 效率的 trade-off

2. **Cache Capacity** (以 token 数为单位):
   - 对于每个数据集，设置为 [20GB, 40GB, 80GB, 160GB, inf]
   - inf 表示无限容量（永不 evict）

3. **Request Ordering** (模拟不同的 cache distance):
   - **Original**: 保持数据集原始顺序
   - **Min Cache Distance**: 同一个 context/session 的 query 紧挨着排列
   - **Max Cache Distance**: 同一个 context/session 的 query 均匀分散
   - **Random Shuffle**: 随机打乱
   - 这对应论文 §5.3.3 的 cache distance 实验

4. **Strategy**: [LRU, LFU, FIFO]

### 输出

对于每个实验配置，输出：
- Page-level cache hit rate (overall)
- Per-request cache hit rate 的分布（mean, p50, p90, p99）
- 需要从 cache load 的 token 数 vs 需要 recompute 的 token 数的比例（load/compute ratio，对应论文 Fig 1 的 x 轴）
- Cache 容量使用情况（peak usage, average usage）
- 最后 RadixTree 的结构统计（比如不同深度节点的访问次数分布）以便分析

## 项目结构

```
kv-cache-sim/
├── README.md
├── requirements.txt          # datasets, tiktoken, matplotlib, pandas, tqdm
├── data/                    # 数据集下载和预处理后的数据
├── src/
│   ├── __init__.py
│   ├── radix_tree.py         # RadixTree 实现（支持 page-level 匹配）
│   ├── cache_simulator.py    # 核心模拟器（支持不同 eviction policy）
│   ├── datasets_loader.py    # 数据集下载和预处理（统一接口）
│   ├── request_generator.py  # 从数据集生成 request stream（支持不同 ordering）
│   └── metrics.py            # 指标计算和统计
│   └── strategies/           # 不同的实验策略的代码 （默认里面实现基于LRU、LFU、FIFO的三个策略，我可能会添加新的自己的策略）
├── experiments/
│   ├── run_page_size.py      # Page size 扫描实验
│   ├── run_ordering.py       # Request ordering 实验
│   ├── run_eviction.py       # Eviction policy 对比实验
│   └── run_all.py            # 跑全部实验组合
├── analysis/
│   └── plot_results.ipynb       # 画图分析脚本
└── results/                  # 实验结果输出目录
```

## 技术要求

1. 代码质量：type hints, docstrings, 合理的抽象
2. 性能：对于大数据集（ShareGPT 200K+ queries），tokenization 结果需要缓存到本地磁盘避免重复计算。能并发的地方并发一下。
3. 进度条：使用 tqdm 显示进度
4. 结果持久化：实验结果保存为 JSON + CSV，方便后续分析
5. 可视化：使用 matplotlib 画出关键图表，特别是：
   - Cache hit rate vs Page size（类似论文 Fig 2）
   - Cache hit rate vs Cache capacity
   - 不同 request ordering 下的 cache hit rate 对比（类似论文 Fig 11）
   - Load/Compute ratio 的 CDF（类似论文 Fig 1 的 x 轴）

