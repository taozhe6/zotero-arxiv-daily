"""
Utility functions for detecting favourite authors and tagging papers
with an `is_favorite` boolean.

All components (scraper, sorter, e-mail builder, etc.) should import
and use these helpers to stay consistent.
"""

from __future__ import annotations
import os
import re
from typing import Sequence, Set


# ------------------------------------------------------------------ #
# Environment-driven favourite-author list
# ------------------------------------------------------------------ #
def _raw_fav_set() -> Set[str]:
    """
    Read the semicolon separated list from the environment variable
    `FAVORITE_AUTHORS` and return a *raw* set of names.
    """
    raw = os.getenv("FAVORITE_AUTHORS", "")
    return {x.strip() for x in raw.split(";") if x.strip()}


def _normalize(text: str) -> str:
    """Alphanumeric only, lower-case – makes fuzzy matching easier."""
    return re.sub(r"\W+", "", text).lower()


_FAV_SET_NORM: Set[str] = {_normalize(name) for name in _raw_fav_set()}


# ------------------------------------------------------------------ #
# Public helpers
# ------------------------------------------------------------------ #
def is_fav_author(name: str) -> bool:
    """
    Return True iff `name` matches any item in the favourite set
    (case-insensitive, punctuation-insensitive *contains* match).
    """
    norm = _normalize(name)
    return any(fav in norm for fav in _FAV_SET_NORM)


def mark_paper(paper) -> None:
    """
    Add / refresh attribute `paper.is_favorite`.

    A paper is considered favourite when *any* author matches
    `FAVORITE_AUTHORS`.
    """
    paper.is_favorite = any(
        is_fav_author(author) for author in getattr(paper, "authors", [])
    )
