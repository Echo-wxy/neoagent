"""Variant-aware search tool (Section 2.5, Equation 5).

``SearchTool`` runs the five-stage pipeline over an offline corpus:
raw retrieval -> variant expansion -> joint BM25 chunk retrieval over the
expanded query set -> semantic filtering -> snippet summarization. The
``BROWSE`` function exposes longer local context without summarization.

Retrieval is joint over the expanded query set Q_exp:
    C_retr = ∪_{q' ∈ Q_exp} BM25_k(q', C_j)         (Equation 5)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .bm25 import BM25, tokenize
from .variants import VariantExpander


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float


def _split_chunks(text: str, max_words: int = 40) -> List[str]:
    """Split a document into coarse chunks for chunk-level retrieval."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], []
    for sent in sentences:
        current.append(sent)
        if sum(len(c.split()) for c in current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def _semantic_overlap(query: str, chunk: str) -> float:
    """Cheap dependency-free stand-in for the Sentence-BERT + cross-encoder
    filtering stage: Jaccard token overlap. Replace with embeddings for a
    higher-fidelity reranker."""
    q = set(tokenize(query))
    c = set(tokenize(chunk))
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


class SearchTool:
    """Offline variant-aware SEARCH and BROWSE tools."""

    def __init__(
        self,
        corpus_path: str | Path,
        expander: VariantExpander | None = None,
        k: int = 20,
        use_variant_expansion: bool = True,
    ):
        records = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        self.docs: List[Dict[str, str]] = records
        self.k = k
        self.use_variant_expansion = use_variant_expansion
        self.expander = expander or VariantExpander()

        # Page-level index for the raw-retrieval stage.
        self._page_texts = [f"{d['title']} {d['text']}" for d in records]
        self._page_bm25 = BM25(self._page_texts)

    # -- SEARCH ----------------------------------------------------------
    def search(self, query: str, top_n: int = 3) -> List[SearchResult]:
        """Five-stage variant-aware search returning summarized snippets."""
        queries = (
            self.expander.expand(query)
            if self.use_variant_expansion
            else [query]
        )

        # Raw retrieval: candidate pages over the union of expanded queries.
        page_scores: Dict[int, float] = {}
        for q in queries:
            for idx, score in self._page_bm25.top_k(q, k=self.k):
                page_scores[idx] = max(page_scores.get(idx, 0.0), score)

        results: List[SearchResult] = []
        for idx in page_scores:
            doc = self.docs[idx]
            chunks = _split_chunks(doc["text"])

            # Joint BM25 chunk retrieval over Q_exp (Equation 5).
            chunk_bm25 = BM25(chunks)
            retr: Dict[int, float] = {}
            for q in queries:
                for cidx, cscore in chunk_bm25.top_k(q, k=self.k):
                    retr[cidx] = max(retr.get(cidx, 0.0), cscore)

            # Semantic filtering -> keep best chunk; summarize to a snippet.
            best_chunk, best_sem = doc["text"], 0.0
            for cidx in retr:
                sem = max(_semantic_overlap(q, chunks[cidx]) for q in queries)
                if sem >= best_sem:
                    best_sem, best_chunk = sem, chunks[cidx]

            results.append(
                SearchResult(
                    url=doc["url"],
                    title=doc["title"],
                    snippet=best_chunk[:240],
                    score=page_scores[idx] + best_sem,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    # -- BROWSE ----------------------------------------------------------
    def browse(self, url: str, query: str) -> str:
        """Return the most relevant long chunk of a page, no summarization."""
        for doc in self.docs:
            if doc["url"] == url:
                chunks = _split_chunks(doc["text"], max_words=60)
                chunks.sort(key=lambda c: _semantic_overlap(query, c), reverse=True)
                return chunks[0] if chunks else doc["text"]
        return ""
