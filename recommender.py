# ----------------------------------------------------------------------
# Re-rank yesterday’s preprints according to similarity with the
# user’s Zotero library (fresh items in Zotero are weighted higher).
# ----------------------------------------------------------------------
from __future__ import annotations
from tqdm import tqdm
from datetime import datetime
from typing import List
import os # 新增导入 os
from concurrent.futures import ThreadPoolExecutor, as_completed # 新增导入并发模块
import torch # 假设 SentenceTransformer 返回 PyTorch tensor，需要导入 torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from loguru import logger # 导入 logger

from paper import PreprintPaper  # unified data model


def rerank_paper(
    candidate: List[PreprintPaper],
    corpus: List[dict],
    model: str = "avsolatorio/GIST-small-Embedding-v0",
) -> List[PreprintPaper]:
    """
    Return `candidate` sorted in descending relevance to the Zotero corpus.

    Scoring strategy
    ----------------
    1. Encode abstracts of both preprints and Zotero items with a
       SentenceTransformer model.
    2. Compute cosine similarity matrix.
    3. Apply logarithmic time-decay weights to Zotero items
       (newer items contribute more).
    4. Aggregate to a single score per preprint and save it to `.score`.
    """
    encoder = SentenceTransformer(model)

    # ---------- build time-decay weights for Zotero items ----------
    corpus_sorted = sorted(
        corpus,
        key=lambda x: datetime.strptime(x["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ"),
        reverse=True,
    )
    # decay = 1 / (1 + np.log10(np.arange(len(corpus_sorted)) + 1))
    decay = np.ones(len(corpus_sorted))
    decay /= decay.sum()  # normalize to 1

    # original code
    # ---------- encode text ----------
    # corpus_emb = encoder.encode(
    #     [item["data"]["abstractNote"] for item in corpus_sorted],
    #     convert_to_tensor=True,
    #     normalize_embeddings=True,
    # )
    # cand_emb = encoder.encode(
    #     [p.summary for p in candidate],
    #     convert_to_tensor=True,
    #     normalize_embeddings=True,
    # )
    corpus_abstracts = [item["data"]["abstractNote"] for item in corpus_sorted]
    
    # 定义一个辅助函数，用于在线程中执行编码任务
    # 这样可以捕获每个任务的异常，并确保结果的独立性
    def _encode_batch_safe(texts_batch, encoder_instance):
        try:
            return encoder_instance.encode(
                texts_batch,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False # 在并行时关闭进度条，避免输出混乱
            )
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            return None # 返回 None 表示该批次失败
    # 根据 GitHub Actions runner 的 CPU 核心数来设置并行度
    # GitHub Actions ubuntu-latest runner 通常有 2 个 CPU 核心
    # 对于 SentenceTransformer 这种底层有 C/CUDA 实现的任务，线程数可以适当多于核心数
    # 经验值：2-4 倍的 CPU 核心数，但为了简单和安全，我们先用 2 倍
    # num_workers = (os.cpu_count() or 2) * 2 # 默认 4 个线程
    num_workers = 2
    # 将语料库摘要分成多个批次
    # 批次大小可以根据实际情况调整，这里取一个经验值，确保每个线程有足够的工作
    # 目标是让每个批次足够大，以摊销 Python 开销，但又不能太大导致内存问题
    # 假设 3947 篇论文，4 个 worker，每个 worker 处理约 1000 篇，分成 10-20 个批次
    batch_size = 512 # 这是一个经验值，可以根据实际情况调整
    
    batches = [corpus_abstracts[i:i + batch_size] for i in range(0, len(corpus_abstracts), batch_size)]
    # 使用 ThreadPoolExecutor 进行并行编码
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交每个批次的编码任务，并记录其原始索引，以便按顺序合并
        future_to_batch_idx = {
            executor.submit(_encode_batch_safe, batch, encoder): i 
            for i, batch in enumerate(batches)
        }
        
        # 收集结果，保持原始批次顺序
        results_map = {}
        for future in tqdm(as_completed(future_to_batch_idx), total=len(batches), desc="Encoding Zotero Corpus"):
            batch_idx = future_to_batch_idx[future]
            try:
                result = future.result()
                if result is not None:
                    results_map[batch_idx] = result
                else:
                    logger.warning(f"Batch {batch_idx} encoding failed, skipping.")
            except Exception as exc:
                logger.error(f"Batch {batch_idx} processing generated an unexpected exception: {exc}")
                results_map[batch_idx] = None # 标记为失败
    # 合并所有成功批次的 embedding
    # 过滤掉可能失败的批次，并按原始批次顺序排序
    valid_embeddings = [results_map[i] for i in sorted(results_map.keys()) if results_map[i] is not None]
    
    if not valid_embeddings:
        logger.error("All Zotero corpus encoding batches failed or returned empty. Cannot proceed with similarity calculation. Returning empty list.")
        return [] # 如果所有批次都失败，则返回空列表，避免后续错误
    # 使用 torch.cat 合并所有批次的 embedding
    corpus_emb = torch.cat(valid_embeddings)
    
    # 候选论文 Embedding (这部分通常很快，因为 candidate 数量少，不需要并行)
    cand_emb = encoder.encode(
        [p.summary for p in candidate],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False # 关闭进度条
    )
    # ---------- similarity & aggregation ----------
    sim_mat = util.cos_sim(cand_emb, corpus_emb)  # shape (n_cand, n_corpus)
    scores = (sim_mat.cpu().numpy() * decay).sum(axis=1) * 10  # scale to ~0-10

    for s, p in zip(scores, candidate):
        p.score = float(s)

    return sorted(candidate, key=lambda x: x.score, reverse=True)
