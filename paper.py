# ----------------------------------------------------------------------
# Fetch yesterday-to-today preprints from four fixed endpoints:
#   • bioRxiv: genetics / genomics / bioinformatics
#   • medRxiv: genetic and genomic medicine
# and merge them into one list (duplicates removed by DOI).
# ----------------------------------------------------------------------
from __future__ import annotations
from loguru import logger
import datetime as _dt
import json as _json
import urllib.error as _uerr
import urllib.parse as _uparse
import urllib.request as _url
from dataclasses import dataclass, field
from typing import List, Optional, Set
from functools import cached_property
import tiktoken
import asyncio # 新增导入 asyncio
from llm import get_llm_client # 修改为 get_llm_client
_UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (GitHub Actions; +https://github.com/DuckLeeyk/zotero-genetics-daily)"
}


def _urlopen_json(url: str) -> dict:
    req = _url.Request(url, headers=_UA_HEADERS)
    try:
        with _url.urlopen(req, timeout=30) as resp:
            return _json.loads(resp.read().decode())
    except _uerr.HTTPError as e:
        logger.error(f"Request failed ({e.code}) → {url}")
        raise


@dataclass
class PreprintPaper:
    title: str
    authors_raw: str
    doi: str
    version: str
    date: str
    category: str
    server: str
    abstract: str

    score: float = 0.0
    code_url: Optional[str] = None

    affiliations: Optional[List[str]] = field(default=None, repr=False)

    is_favorite: bool = field(default=False, repr=False, compare=False)
    # 新增一个属性来存储 TLDR 结果
    tldr_content: str = field(default="", repr=False) # 默认值可以为空字符串或摘要
    @property
    def authors(self) -> List[str]:
        return [a.strip() for a in self.authors_raw.split(";") if a.strip()]

    @property
    def summary(self) -> str:
        return self.abstract
    
    # 将 tldr 属性改为异步方法
    async def get_tldr(self) -> str: # 修改为异步方法
        llm_instance = get_llm_client() # 获取全局 LLM 客户端实例
        
        prompt_template = """Given the title and abstract of a paper, generate a one-sentence TLDR summary in __LANG__:
        
        Title: __TITLE__
        Abstract: __ABSTRACT__
        """
        
        prompt = prompt_template.replace('__LANG__', llm_instance.lang)
        prompt = prompt.replace('__TITLE__', self.title)
        prompt = prompt.replace('__ABSTRACT__', self.abstract)
        
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]
        prompt = enc.decode(prompt_tokens)
        tldr_result = self.abstract
        try:
            # 调用异步的 generate_tldr 方法
            tldr_content = await llm_instance.generate_tldr( 
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            logger.debug(f"Generated TLDR for {self.doi} in {llm_instance.lang}")
            self.tldr_content = tldr_result
            return tldr_content
        except Exception as e:
            logger.error(f"Failed to generate TLDR for {self.doi}: {e}. Falling back to abstract.")
            self.tldr_content = tldr_result
            return tldr_result
        
    @property
    def paper_id(self) -> str:
        return self.doi

    @property
    def url(self) -> str:
        return f"https://www.{self.server}.org/content/{self.doi}v{self.version}"

    @property
    def pdf_url(self) -> str:
        return self.url + ".full.pdf"


def _build_endpoints() -> List[str]:
    today = _dt.date.today()
    yesterday = today - _dt.timedelta(days=1)
    start, end = yesterday.isoformat(), today.isoformat()

    categories_bio = ["genetics", "genomics", "bioinformatics"]
    eps: List[str] = [
        f"https://api.biorxiv.org/details/biorxiv/{start}/{end}?category={cat}"
        for cat in categories_bio
    ]

    cat_med = _uparse.quote("genetic and genomic medicine")
    eps.append(
        f"https://api.biorxiv.org/details/medrxiv/{start}/{end}?category={cat_med}"
    )
    return eps


def fetch_today_papers(
    *_, today: Optional[_dt.date] = None
) -> List[PreprintPaper]:
    if today is not None:
        delta = today - _dt.timedelta(days=1)
        start, end = delta.isoformat(), today.isoformat()
        endpoints = [
            f"https://api.biorxiv.org/details/biorxiv/{start}/{end}?category=genetics",
            f"https://api.biorxiv.org/details/biorxiv/{start}/{end}?category=genomics",
            f"https://api.biorxiv.org/details/biorxiv/{start}/{end}?category=bioinformatics",
            "https://api.biorxiv.org/details/medrxiv/"
            f"{start}/{end}?category=" + _uparse.quote("genetic and genomic medicine"),
        ]
    else:
        endpoints = _build_endpoints()

    logger.info("Fetching preprints from fixed endpoints …")

    seen: Set[str] = set()
    papers: List[PreprintPaper] = []

    for url in endpoints:
        server = "biorxiv" if "/biorxiv/" in url else "medrxiv"

        # -------- NEW: extract category for logging --------
        parsed = _uparse.urlparse(url)
        cat = _uparse.unquote_plus(_uparse.parse_qs(parsed.query).get("category", [""])[0])
        # ----------------------------------------------------

        try:
            payload = _urlopen_json(url)
        except Exception:
            continue

        col = payload.get("collection", [])
        for item in col:
            doi = item["doi"]
            if doi in seen:
                continue
            seen.add(doi)

            # Extract the corresponding institution field into a list
            corr_inst = item.get("author_corresponding_institution", "").strip()
            if not corr_inst:
                corr_inst = "Unknown Affiliation"
            affiliations = [corr_inst]

            papers.append(
                PreprintPaper(
                    title=item["title"],
                    authors_raw=item["authors"],
                    doi=doi,
                    version=str(item["version"]),
                    date=item["date"],
                    category=item["category"],
                    server=server,
                    abstract=item["abstract"],
                    affiliations=affiliations,   # <-- new
                )
            )

        logger.debug(f"{server}: {len(col)} items fetched ({cat}).")

    logger.success(f"{len(papers)} unique preprints collected.")
    return papers


if __name__ == "__main__":
    for p in fetch_today_papers()[:5]:
        aff = p.affiliations[0] if p.affiliations else "Unknown Affiliation"
        print(f"[{p.server}] {p.date}  {p.title}  — Affiliation: {aff}")
