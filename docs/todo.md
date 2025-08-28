# Zotero Genetics Daily - 后续优化计划

## 1. 优化 Zotero 语料库文献的权重策略

**当前问题：**
目前的权重策略 (`1 / (1 + np.log10(rank))`) 导致最近添加到 Zotero 的文献权重最高，而较旧的文献权重迅速衰减。这对于研究方向可能发生转变的用户（例如从 Meta 分析转向 Genetics 论文）来说，可能不合理，因为它过度强调了历史文献的“新旧”程度，而忽略了其潜在的长期相关性或新方向的兴趣。

**用户反馈：**
用户指出，这种加权方式可能导致推荐结果过度偏向旧的研究方向，而无法有效捕捉到新的兴趣点。

**优化方向（待讨论和实现）：**

*   **A. 不加权 (Equal Weighting)：**
    *   所有 Zotero 语料库中的文献都具有相同的权重。
    *   **优点：** 简单，确保所有历史文献都能平等地贡献相似度。
    *   **缺点：** 可能无法体现用户对近期研究的偏好。
    *   **实现思路：** 将 `decay` 数组设置为所有元素都为 `1 / len(corpus_sorted)`。

*   **B. 线性衰减 (Linear Decay)：**
    *   权重随时间线性衰减，而不是对数衰减。衰减速度更平缓。
    *   **优点：** 比对数衰减更温和，仍能体现时间因素，但不会过快地降低旧文献的影响。
    *   **实现思路：** `decay = np.linspace(max_weight, min_weight, len(corpus_sorted))`，然后归一化。

*   **C. 区分日期，不区分时间 (Daily Batch Weighting)：**
    *   同一天添加到 Zotero 的所有文献具有相同的权重。不同日期的文献权重不同。
    *   **优点：** 更符合用户“批次”添加文献的习惯，避免了微小时间戳差异带来的不必要权重差异。能更好地反映用户在某个时间段内的兴趣。
    *   **实现思路：**
        1.  按 `dateAdded` 的日期部分对文献进行分组。
        2.  对每个日期组分配一个权重（例如，最近的日期组权重最高，然后线性或对数衰减）。
        3.  组内所有文献共享该组的权重。

**优先级：** 中等（在核心功能稳定运行后，根据用户反馈和实际效果进行迭代）。

## 2. 优化网络请求性能 (异步化)

**当前问题：**
`main.py` 中 Zotero API 和 BioRxiv/MedRxiv API 的数据抓取是同步进行的，可能存在 I/O 阻塞，尤其是在网络延迟较高时。

**优化方向：**
引入 `async/await` 异步编程，使用 `aiohttp` 等异步 HTTP 客户端并行或异步地抓取数据。

**优先级：** 低（在计算瓶颈解决后，如果网络请求成为新的瓶要点，再进行考虑）。


# TODO List for Preprint Recommender
## LLM API 优化
### 1. 异步调用 OpenAI 兼容 API
**目标：** 提高生成 TLDR 的效率，尤其是在需要处理大量论文时。
**当前状态：** `LLM.generate` 方法目前是同步调用，导致 TLDR 生成过程串行执行。
**实现思路：**
*   修改 `llm.py`：
    *   将 `openai.OpenAI` 客户端替换为 `openai.AsyncOpenAI`。
    *   将 `LLM.generate` 方法改为 `async def`。
    *   在 `generate` 方法内部，使用 `await self.llm.chat.completions.create(...)` 进行异步调用。
    *   引入 `backoff` 库（`pip install backoff`）来自动处理 `openai.RateLimitError` 和其他网络异常，实现指数退避重试机制。
*   修改 `main.py`：
    *   将主执行流封装在一个 `async def main_flow()` 函数中。
    *   在脚本入口点使用 `asyncio.run(main_flow())` 运行。
    *   在需要生成 TLDR 的地方，收集所有 `llm_instance.generate(messages)` 协程任务到一个列表中。
    *   使用 `await asyncio.gather(*tldr_tasks, return_exceptions=True)` 并行执行这些任务，并确保结果与原始论文顺序对应。
    *   使用 `tqdm` 包装 `asyncio.gather` 以显示进度条。
*   **控制方式：** 可以通过一个新的环境变量（例如 `USE_ASYNC_LLM=true`）来控制是否启用异步调用。如果未设置或为 `false`，则回退到同步调用。
### 2. LLM API Key 池
**目标：** 绕过单个 API Key 的速率限制（例如 Gemini 的 10/minute），通过轮询多个 API Key 来提高吞吐量。
**当前状态：** `LLM` 类目前只支持单个 API Key。
**实现思路：**
*   修改 `llm.py`：
    *   `LLM` 类不再直接持有单个 `AsyncOpenAI` 客户端，而是维护一个 `AsyncOpenAI` 客户端的列表（池），每个客户端对应一个 API Key。
    *   `set_global_llm` 应该能够接收一个 API Key 列表（例如，通过逗号分隔的环境变量 `OPENAI_API_KEYS="key1,key2,key3"`）。
    *   在 `LLM.generate` 方法中，实现一个简单的轮询机制，从池中选择一个可用的客户端。
    *   需要考虑如何处理某个 Key 达到速率限制的情况（例如，暂时将其从可用池中移除一段时间）。
*   **控制方式：** 可以通过环境变量（例如 `OPENAI_API_KEYS` 包含多个 Key 时自动启用）来控制是否启用 Key 池。


## 待办事项
### 1. 优化 `rerank_paper` 函数中的 Zotero Corpus 编码性能 (多进程)
**背景：**
当前的 `rerank_paper` 函数使用 `ThreadPoolExecutor` 对 Zotero Corpus 进行编码，但在 GitHub Actions 的 2 核 CPU 环境下，性能提升不明显（仅从 601 秒缩短到 565 秒）。这可能是由于 Python GIL 的限制以及 GitHub Actions 资源的调度开销。
**目标：**
通过使用 `ProcessPoolExecutor` 绕过 GIL，充分利用多核 CPU，进一步显著缩短 Zotero Corpus 编码阶段的耗时。
**实施方案：**
将 `rerank_paper` 函数中对 `corpus_abstracts` 进行编码的部分，从 `ThreadPoolExecutor` 切换到 `ProcessPoolExecutor`。
**具体修改点：**
1.  **导入 `ProcessPoolExecutor`：** 将 `from concurrent.futures import ThreadPoolExecutor, as_completed` 修改为 `from concurrent.futures import ProcessPoolExecutor, as_completed`。
2.  **替换 `ThreadPoolExecutor`：** 将 `with ThreadPoolExecutor(...)` 修改为 `with ProcessPoolExecutor(...)`。
3.  **调整 `num_workers`：** 将 `num_workers` 设置为 `os.cpu_count() or 2`，即与 GitHub Actions runner 的物理 CPU 核心数匹配（通常为 2）。
4.  **模型加载策略调整：**
    *   由于 `SentenceTransformer` 模型实例不能直接在进程间序列化传递，需要在每个子进程内部重新加载模型。
    *   因此，`_encode_batch_safe` 辅助函数需要修改为接收 `model_name` (字符串) 而不是 `encoder_instance`。
    *   在 `_encode_batch_safe` 函数内部，使用 `local_encoder = SentenceTransformer(model_name)` 重新加载模型。
    *   主进程也需要一个 `encoder` 实例来编码 `candidate` 论文，因此在 `cand_emb` 编码前，也需要 `main_process_encoder = SentenceTransformer(model)`。
**潜在风险/注意事项：**
*   **模型重复加载开销：** 每个子进程都会重新加载模型。对于 `avsolatorio/GIST-small-Embedding-v0` 这种小模型，开销可能可接受，但对于大模型可能成为新的瓶颈。
*   **内存消耗：** 每个进程都有独立的内存空间，多进程可能会比多线程消耗更多内存。在 GitHub Actions 这种资源受限的环境中，需要监控内存使用情况，避免 OOM (Out Of Memory) 错误。
*   **启动时间：** 启动进程的开销通常大于启动线程。
**测试计划：**
1.  实施上述代码修改。
2.  在 GitHub Actions 上运行，并记录 `Zotero Corpus 编码` 阶段的耗时。
3.  与当前 565 秒的耗时进行比较，评估性能提升。
4.  监控 GitHub Actions 的内存使用情况。
**预期效果：**
如果 GitHub Actions 具有 2 个物理核心且模型加载开销可控，预计 `Zotero Corpus 编码` 阶段的耗时将显著缩短，可能接近 2 倍的加速。
