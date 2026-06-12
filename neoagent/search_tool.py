"""Variant-aware search tool (Section 2.5, Equation 5).

Five-stage SEARCH over a pluggable backend:

  1. raw retrieval            (search_backend.raw_search)
  2. variant expansion        (variant_expander.expand)  -> Q_exp
  3. joint BM25 chunk retrieval over Q_exp:
         C_retr = U_{q' in Q_exp} BM25_k(q', C_j)          (Equation 5)
  4. semantic filtering + cross-encoder rerank -> top-3 chunks
  5. LLM summarization of the kept chunks into a compact snippet (optional)

BROWSE returns longer local context for a single page without summarization.

The tool also exposes the four core functions named in the paper:
``bm25``, ``embed_retrieve``, ``rerank`` and ``expand_variants``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .bm25 import BM25
from .llm import LLMClient, Message
from .retrieval import CrossEncoderReranker, DenseRetriever
from .search_backend import SearchBackend


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float


def _split_chunks(text: str, max_words: int = 40) -> List[str]:
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


class SearchTool:
    """Variant-aware SEARCH / BROWSE over a :class:`SearchBackend`.

    Parameters
    ----------
    backend:
        Raw-retrieval source (offline corpus or live web search).
    expander:
        Object exposing ``expand(query) -> list[str]`` (learned or rule-based).
    retriever, reranker:
        Semantic filter and cross-encoder for stage 4. Defaults degrade to a
        token-overlap stand-in when the models are unavailable.
    summarizer:
        Optional :class:`LLMClient` for stage 5. When ``None``, the kept chunk
        is returned verbatim (no summarization).
    k, top_n, use_variant_expansion:
        Chunk retrieval depth, number of returned results, and the ablation
        switch used to disable variant expansion ("w/o VarExp").
    """

    def __init__(
        self,
        backend: SearchBackend,
        expander,
        retriever: Optional[DenseRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        summarizer: Optional[LLMClient] = None,
        k: int = 20,
        top_n: int = 3,
        use_variant_expansion: bool = True,
    ):
        self.backend = backend
        self.expander = expander
        self.retriever = retriever or DenseRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.summarizer = summarizer
        self.k = k
        self.top_n = top_n
        self.use_variant_expansion = use_variant_expansion

    # -- four core functions (paper, Section 2.5) ------------------------
    def expand_variants(self, snippet: str) -> List[str]:
        """Predict plausible alternative surface forms; ``[]`` if none confident."""
        if not self.use_variant_expansion:
            return []
        return self.expander.expand(snippet)[1:]

    @staticmethod
    def bm25(query: str, corpus: Sequence[str], k: int = 20):
        """Top-k chunks by BM25 score."""
        return BM25(list(corpus)).top_k(query, k=k)

    def embed_retrieve(self, query: str, corpus: Sequence[str], top_k: int = 3):
        """Semantically close chunks above the embedding similarity ordering."""
        return self.retriever.filter(query, corpus, top_k=top_k)

    def rerank(self, query: str, candidates: Sequence[str], top_k: int = 3):
        """Select the highest-value candidates via the cross-encoder."""
        return self.reranker.rerank(query, candidates, top_k=top_k)

    # -- SEARCH ----------------------------------------------------------
    def _query_set(self, query: str) -> List[str]:
        if not self.use_variant_expansion:
            return [query]
        return self.expander.expand(query)

    def search(self, query: str) -> List[SearchResult]:
        queries = self._query_set(query)

        # Stage 1: raw retrieval over the union of expanded queries.
        pages = {}
        for q in queries:
            for r in self.backend.raw_search(q, k=self.k):
                if r.url not in pages or r.score > pages[r.url].score:
                    pages[r.url] = r

        results: List[SearchResult] = []
        for raw in pages.values():
            page_text = self.backend.fetch(raw.url) or raw.snippet
            chunks = _split_chunks(page_text)

            # Stage 3: joint BM25 chunk retrieval over Q_exp (Equation 5).
            retr_idx = {}
            for q in queries:
                for cidx, cscore in self.bm25(q, chunks, k=self.k):
                    retr_idx[cidx] = max(retr_idx.get(cidx, 0.0), cscore)
            if not retr_idx:
                continue
            cand_chunks = [chunks[i] for i in retr_idx]

            # Stage 4: dense filter -> cross-encoder rerank -> top chunks.
            filtered = self.embed_retrieve(query, cand_chunks, top_k=max(self.top_n, 3))
            shortlist = [cand_chunks[i] for i, _ in filtered] or cand_chunks
            reranked = self.rerank(query, shortlist, top_k=self.top_n)
            kept = [shortlist[i] for i, _ in reranked]

            # Stage 5: LLM summarization (optional).
            snippet = self._summarize(query, kept) if self.summarizer else " ".join(kept)[:300]
            results.append(
                SearchResult(raw.url, raw.title, snippet, raw.score)
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: self.top_n]

    def _summarize(self, query: str, chunks: Sequence[str]) -> str:
        context = "\n".join(f"- {c}" for c in chunks)
        msg = [
            Message("system", "Summarize the evidence relevant to the query in 1-2 sentences. "
                              "Keep answer-bearing facts; drop boilerplate."),
            Message("user", f"Query: {query}\nEvidence:\n{context}"),
        ]
        return self.summarizer.chat(msg, max_tokens=160)

    # -- BROWSE ----------------------------------------------------------
    def browse(self, url: str, query: str) -> str:
        """Return the most relevant long chunk of a page, no summarization."""
        text = self.backend.fetch(url)
        if not text:
            return ""
        chunks = _split_chunks(text, max_words=80)
        scored = self.embed_retrieve(query, chunks, top_k=1)
        return chunks[scored[0][0]] if scored else text
