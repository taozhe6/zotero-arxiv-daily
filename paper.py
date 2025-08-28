# ----------------------------------------------------------------------
# Fetch yesterday-to-today preprints from four fixed endpoints:
#   • bioRxiv: genetics / genomics / bioinformatics
#   • medRxiv: genetic and genomic medicine
# and merge them into one list (duplicates removed by DOI).
# ----------------------------------------------------------------------
from __future__ import annotations

import datetime as _dt
import json as _json
import urllib.error as _uerr
import urllib.parse as _uparse
import urllib.request as _url
from dataclasses import dataclass, field
from typing import List, Optional, Set

from loguru import logger

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
    _tldr: str = field(default="", repr=False)
    @property
    def authors(self) -> List[str]:
        return [a.strip() for a in self.authors_raw.split(";") if a.strip()]

    @property
    def summary(self) -> str:
        return self.abstract
    
    @property
    def tldr(self) -> str:
        # 如果_tldr有内容，就返回_tldr；否则返回原始摘要
        return self._tldr if self._tldr else self.abstract

    @tldr.setter # <-- 新增这个setter，允许外部修改tldr
    def tldr(self, value: str):
        self._tldr = value
        
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
