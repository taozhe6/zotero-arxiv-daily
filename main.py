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
from llm import set_global_llm

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
        else:
            env_val = kwargs.get("type", str)(env_val)
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

# Misc
parser.add_argument("--debug", action="store_true", help="Verbose logging")

args = parser.parse_args()
logger.debug(f"Parsed language argument: {args.language}")
# ============================  Logging  ============================== #
logger.remove()
logger.add(sys.stdout, level="DEBUG" if args.debug else "INFO")

if args.use_llm_api and args.openai_api_key is None:
    parser.error("--use_llm_api requires --openai_api_keys (comma-separated)")

fav_author_set = _load_favorite_authors(args.favorite_authors)

# =============================  Flow  ================================ #
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
    papers = rerank_paper(papers, corpus)

    # Mark favorites and resort BEFORE applying max_paper_num
    papers = mark_and_sort(papers, fav_author_set)

    if args.max_paper_num != -1:
        papers = papers[: args.max_paper_num]

    # Configure the global LLM
    if args.use_llm_api:
        logger.info("Generating TLDR with cloud LLM …")
        set_global_llm(
            api_key=args.openai_api_key,
            base_url=args.openai_api_base,
            model=args.model_name,
            lang=args.language,
        )
    else:
        logger.info("Generating TLDR with local LLM …")
        set_global_llm(lang=args.language)
    
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
logger.success(
    "E-mail sent! If nothing arrives, check the spam folder or SMTP settings."
)
