# ----------------------------------------------------------------------
# Re-rank yesterday’s preprints according to similarity with the
# user’s Zotero library (fresh items in Zotero are weighted higher).
# ----------------------------------------------------------------------
from __future__ import annotations

from datetime import datetime
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer, util

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
    decay = 1 / (1 + np.log10(np.arange(len(corpus_sorted)) + 1))
    decay /= decay.sum()  # normalize to 1

    # ---------- encode text ----------
    corpus_emb = encoder.encode(
        [item["data"]["abstractNote"] for item in corpus_sorted],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    cand_emb = encoder.encode(
        [p.summary for p in candidate],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    # ---------- similarity & aggregation ----------
    sim_mat = util.cos_sim(cand_emb, corpus_emb)  # shape (n_cand, n_corpus)
    scores = (sim_mat.cpu().numpy() * decay).sum(axis=1) * 10  # scale to ~0-10

    for s, p in zip(scores, candidate):
        p.score = float(s)

    return sorted(candidate, key=lambda x: x.score, reverse=True)
