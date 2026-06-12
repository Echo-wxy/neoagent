"""Semantic filtering and reranking for the search tool (Section 2.5).

The paper's SEARCH pipeline, after joint BM25 chunk retrieval over the expanded
query set, applies a **Sentence-BERT** embedding filter followed by a
**cross-encoder reranker** to compress candidate chunks down to a small,
high-precision evidence set (top-3).

This module provides those two stages with a real implementation when
``sentence-transformers`` is installed, and a transparent token-overlap
fallback otherwise (clearly marked, for smoke tests only).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .bm25 import tokenize

# Default models (override via configs/search_tool.yaml).
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _token_overlap(query: str, text: str) -> float:
    q, c = set(tokenize(query)), set(tokenize(text))
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


class DenseRetriever:
    """Sentence-embedding similarity filter (with token-overlap fallback)."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._fallback = False

    def _ensure(self):
        if self._model is not None or self._fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception:
            # No model / library available -> degrade to token overlap.
            self._fallback = True

    def filter(self, query: str, chunks: Sequence[str], top_k: int = 3) -> List[Tuple[int, float]]:
        """Return ``[(chunk_index, score), ...]`` for the ``top_k`` chunks."""
        self._ensure()
        if self._fallback:
            scored = [(i, _token_overlap(query, c)) for i, c in enumerate(chunks)]
        else:
            from sentence_transformers import util

            q_emb = self._model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
            c_emb = self._model.encode(list(chunks), convert_to_tensor=True, normalize_embeddings=True)
            sims = util.cos_sim(q_emb, c_emb)[0].tolist()
            scored = list(enumerate(sims))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker over (query, candidate) pairs."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._fallback = False

    def _ensure(self):
        if self._model is not None or self._fallback:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        except Exception:
            self._fallback = True

    def rerank(self, query: str, candidates: Sequence[str], top_k: int = 3) -> List[Tuple[int, float]]:
        """Return ``[(candidate_index, score), ...]`` best-first."""
        self._ensure()
        if self._fallback:
            scored = [(i, _token_overlap(query, c)) for i, c in enumerate(candidates)]
        else:
            pairs = [(query, c) for c in candidates]
            scores = self._model.predict(pairs).tolist()
            scored = list(enumerate(scores))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]
