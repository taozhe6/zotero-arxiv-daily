"""
Recommend yesterday’s new preprints (bioRxiv + medRxiv)
based on your Zotero library and send them by e-mail.
"""

from __future__ import annotations

import argparse
import os
import sys
from tempfile import mkstemp
from typing import List, Set
import asyncio # 新增导入 asyncio
from dotenv import load_dotenv
from simple_key_pool import SimpleKeyPool
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from gitignore_parser import parse_gitignore
from loguru import logger
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm.asyncio import tqdm as async_tqdm

from fav_utils import mark_paper
from paper import PreprintPaper, fetch_today_papers
from llm import set_global_llm_client, close_global_key_pool, GLOBAL_KEY_POOL

# =============================  Zotero  ============================== #
def get_zotero_corpus(user_id: str, api_key: str) -> list[dict]:
    """Download the complete Zotero library (conference/journal/preprint only)."""
    zot = zotero.Zotero(user_id, "user", api_key)

    collections = zot.everything(zot.collections())
    collections = {c["key"]: c for c in collections}

    corpus = zot.everything(
        zot.items(itemType="conferencePaper || journalArticle || preprint")
    )
    corpus = [c for c in corpus if c["data"]["abstractNote"]]

    def _path_of(col_key: str) -> str:
        parent = collections[col_key]["data"]["parentCollection"]
        if parent:
            return _path_of(parent) + "/" + collections[col_key]["data"]["name"]
        return collections[col_key]["data"]["name"]

    for c in corpus:
        c["paths"] = [_path_of(col) for col in c["data"]["collections"]]
    return corpus


def filter_corpus(corpus: list[dict], pattern: str) -> list[dict]:
    """Exclude items whose collection path matches the gitignore-style pattern."""
    _, tmpfile = mkstemp()
    with open(tmpfile, "w") as f:
        f.write(pattern)
    matcher = parse_gitignore(tmpfile, base_dir="./")
    kept = [c for c in corpus if not any(matcher(p) for p in c["paths"])]
    os.remove(tmpfile)
    return kept


# ====================  Favorite-author utilities  ==================== #
def _load_favorite_authors(raw: str) -> Set[str]:
    """
    Convert a semicolon-separated string into a lower-cased set.
    Empty elements are discarded.
    """
    return {x.strip().lower() for x in raw.split(";") if x.strip()}


def mark_and_sort(papers: List[PreprintPaper], fav_authors: Set[str]) -> List[PreprintPaper]:
    """
    Mark papers containing favorite authors and move them to the front.
    Secondary key remains the relevance score (descending).
    """
    for p in papers:
        mark_paper(p)
        
    papers.sort(key=lambda x: (not x.is_favorite, -x.score))
    return papers


# ==========================  Preprint fetch  ========================= #
def get_preprints(debug: bool = False) -> list[PreprintPaper]:
    """
    Return yesterday’s preprints from bioRxiv & medRxiv.
    With debug=True only the first 5 items are kept.
    """
    papers = fetch_today_papers()
    if debug:
        logger.debug("Debug mode: returning only 5 example papers.")
        papers = papers[:5]
    return papers


# ==========================  CLI definition  ========================= #
parser = argparse.ArgumentParser(
    description="bioRxiv/medRxiv -- Zotero recommender and mailer"
)


def add_argument(*args, **kwargs):
    """
    Allow every CLI argument to be overridden by an env var of the same
    uppercase name. E.g. --smtp_server ←→ SMTP_SERVER
    """
    dest_name = kwargs.get("dest", args[-1].lstrip("-").replace("-", "_"))

    def _env(key: str, default=None):
        val = os.environ.get(key)
        return default if val in [None, ""] else val

    parser.add_argument(*args, **kwargs)
    env_val = _env(dest_name.upper())
    if env_val is not None:
        if kwargs.get("type") is bool:
            env_val = env_val.lower() in ["1", "true"]
        elif kwargs.get("type") is int: # <-- 新增：处理 int 类型
            try:
                env_val = int(env_val)
            except ValueError:
                logger.warning(f"Environment variable {dest_name.upper()} is not a valid integer. Using default.")
                env_val = None # 设为 None 让 argparse 使用默认值
        else:
            env_val = kwargs.get("type", str)(env_val)
        
        if env_val is not None: # 只有当 env_val 有效时才设置默认值
            parser.set_defaults(**{dest_name: env_val})


# Zotero
add_argument("--zotero_id", type=str, help="Zotero user ID")
add_argument("--zotero_key", type=str, help="Zotero API key")
add_argument(
    "--zotero_ignore", type=str, help="Zotero collection ignore pattern (gitignore style)"
)
add_argument(
    "--send_empty",
    type=bool,
    default=False,
    help="Send an empty e-mail even when there is no new paper",
)
add_argument(
    "--max_paper_num",
    type=int,
    default=100,
    help="Maximum number of papers shown in the e-mail (-1 for unlimited)",
)

# SMTP / e-mail
add_argument("--smtp_server", type=str, help="SMTP server")
add_argument("--smtp_port", type=int, help="SMTP port")
add_argument("--sender", type=str, help="Sender e-mail")
add_argument("--receiver", type=str, help="Receiver e-mail")
add_argument("--sender_password", type=str, help="Sender password")

# LLM
add_argument(
    "--use_llm_api",
    type=bool,
    default=False,
    help="Use a cloud LLM (OpenAI-compatible API) for TLDR generation",
)
# 修改为支持多个 API Key，通过环境变量传入
add_argument("--openai_api_keys", type=str, default=None, help="Comma-separated OpenAI API Keys (e.g., sk-key1,sk-key2)")
add_argument(
    "--openai_api_base",
    type=str,
    default="https://api.openai.com/v1",
    help="OpenAI Base URL",
)
add_argument("--model_name", type=str, default="gpt-4o", help="Model name")
add_argument("--language", type=str, default="English", help="TLDR language")
# 新增密钥池相关参数
add_argument("--llm_key_pool_blacklist_threshold", type=int, default=3, help="LLM key pool blacklist threshold")
add_argument("--llm_key_pool_recovery_interval", type=int, default=300, help="LLM key pool recovery interval in seconds")
# Favorite authors
add_argument(
    "--favorite_authors",
    type=str,
    default="",
    help="Semicolon-separated list of favorite author names",
)
add_argument("--llm_key_pool_rpm_limit", type=int, default=10, help="LLM key pool RPM limit per key.")
add_argument("--concurrency_limit", type=int, default=2, help="Limit the number of concurrent API requests.")
# Misc
parser.add_argument("--debug", action="store_true", help="Verbose logging")

args = parser.parse_args()
logger.debug(f"Parsed language argument: {args.language}")
# ============================  Logging  ============================== #
logger.remove()
logger.add(sys.stdout, level="DEBUG" if args.debug else "INFO")

if args.use_llm_api and args.openai_api_keys is None:
    parser.error("--use_llm_api requires --openai_api_keys (comma-separated)")

fav_author_set = _load_favorite_authors(args.favorite_authors)

# =============================  Flow  ================================ #
async def main_async_flow(): # 将主逻辑封装为异步函数
    global GLOBAL_KEY_POOL # 声明使用全局密钥池
    logger.info("Downloading Zotero library …")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"{len(corpus)} items found")
    if args.zotero_ignore:
        logger.info("Applying ignore rules …")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"{len(corpus)} items remain after filtering")
    logger.info("Fetching new preprints …")
    papers = get_preprints(args.debug)
    if len(papers) == 0:
        logger.info("No new preprints found")
        if not args.send_empty:
            sys.exit(0)
    else:
        logger.info("Computing similarity scores and re-ranking …")
        # 恢复 rerank_paper 为同步调用，移除 ThreadPoolExecutor
        # 如果你之前已经修改了 rerank_paper 为同步，这里无需额外修改
        # 如果没有，请参考我之前给出的“方案一：测试禁用并发”的代码
        papers = rerank_paper(papers, corpus) 
        # Mark favorites and resort BEFORE applying max_paper_num
        papers = mark_and_sort(papers, fav_author_set)
        if args.max_paper_num != -1:
            papers = papers[: args.max_paper_num]
        # 配置全局 LLM 客户端和密钥池
        if args.use_llm_api:
            logger.info("Generating TLDR with cloud LLM …")
            # 初始化密钥池
            api_keys = [k.strip() for k in args.openai_api_keys.split(',') if k.strip()]
            if len(api_keys) > 1: # 如果有多个密钥，则使用密钥池
                GLOBAL_KEY_POOL = SimpleKeyPool(
                    keys=api_keys,
                    blacklist_threshold=args.llm_key_pool_blacklist_threshold,
                    recovery_interval_seconds=args.llm_key_pool_recovery_interval,
                    rpm_limit=args.llm_key_pool_rpm_limit # <-- 传递 RPM 限制
                )
                set_global_llm_client(
                    base_url=args.openai_api_base,
                    model=args.model_name,
                    lang=args.language,
                    key_pool=GLOBAL_KEY_POOL # 将密钥池实例传递给 LLM 客户端
                )
            else: # 只有一个密钥，兼容旧逻辑，不使用密钥池
                set_global_llm_client(
                    api_key=api_keys[0],
                    base_url=args.openai_api_base,
                    model=args.model_name,
                    lang=args.language,
                    key_pool=None # 不使用密钥池
                )
            
            # 使用 Semaphore 控制并发
            semaphore = asyncio.Semaphore(args.concurrency_limit)
            logger.info(f"Generating TLDRs with concurrency limit: {args.concurrency_limit}")
            
            async def get_tldr_with_semaphore(paper: PreprintPaper):
                async with semaphore:
                    return await paper.get_tldr()
            
            tldr_tasks = [get_tldr_with_semaphore(paper) for paper in papers]
            tldr_results = await async_tqdm.gather(*tldr_tasks, desc="Generating TLDRs concurrently")
            for paper, tldr_text in zip(papers, tldr_results):
                paper.tldr_content = tldr_text
        else:
            logger.info("Generating TLDR with local LLM …")
            set_global_llm_client(lang=args.language)
            # 如果是本地 LLM，虽然 get_tldr 是 async 方法，但内部是同步调用
            # 仍然可以并发执行，但不会有 I/O 并行优势
            tldr_tasks = [paper.get_tldr() for paper in papers]
            tldr_results = await async_tqdm.gather(*tldr_tasks, desc="Generating TLDRs with local LLM")
            for paper, tldr_text in zip(papers, tldr_results):
                paper.tldr_content = tldr_text
    # =======================  Build and send e-mail  ====================== #
    logger.info("Rendering e-mail …")
    html = render_email(papers)
    logger.info("Sending e-mail …")
    send_email(
        args.sender,
        args.receiver,
        args.sender_password,
        args.smtp_server,
        args.smtp_port,
        html,
    )
    # --- Start of Performance Metrics Calculation ---
    successful_tldrs = 0
    for p in papers:
        # 假设原文摘要不以中文开头，而成功生成的TLDR是中文
        # 这是一个简单的启发式判断，您可以根据实际情况调整
        if p.tldr_content != p.abstract:
            successful_tldrs += 1
    
    total_papers = len(papers)
    success_rate = (successful_tldrs / total_papers) * 100 if total_papers > 0 else 0
    
    # 假设 tldr_tasks 是在 if args.use_llm_api 块中定义的
    # 我们需要获取总执行时间。一个简单的方法是在生成TLDR前后记录时间
    # （为了不让您修改太多地方，我们这里用一个近似值）
    # 假设每个请求的基线时间是 6 秒 (10 RPM)
    baseline_time = total_papers * (60 / args.llm_key_pool_rpm_limit)
    
    # 从日志中获取实际执行时间是一个复杂的操作，
    # 但我们可以通过tqdm的输出来估算。
    # 为了简单起见，我们直接在日志中打印出可用于计算的数据。
    # 我们需要找到 tldr_results = await ... 这一行来计算实际时间
    
    # 为了避免大幅修改，我们先打印核心数据
    logger.info("==================== PERFORMANCE METRICS ====================")
    logger.info(f"Total Papers Processed: {total_papers}")
    logger.info(f"Successful TLDRs Generated: {successful_tldrs}")
    logger.info(f"Success Rate: {success_rate:.2f}%")
    
    # 构建综合指标
    # Score = (Success Rate / 100) * (Baseline Time / Actual Time)
    # 由于我们在这里拿不到精确的 Actual Time，我们先打印一个提示
    logger.info("To calculate Performance Score, please find the total time for 'Generating TLDRs concurrently' in the logs above.")
    logger.info("Formula: Score = (Success Rate / 100) * (Baseline Time / Actual Time)")
    logger.info(f"For this run, Baseline Time would be approx: {baseline_time:.2f} seconds.")
    logger.info("============================================================")
    # --- End of Performance Metrics Calculation ---
    logger.success(
        "E-mail sent! If nothing arrives, check the spam folder or SMTP settings."
    )
    # 关闭全局密钥池（如果存在）
    await close_global_key_pool()
if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main_async_flow())
