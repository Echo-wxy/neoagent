"""Raw-retrieval backends for the search tool.

The variant-aware SEARCH pipeline begins with a *raw retrieval* stage that
fetches initial results from a search engine, then augments them with variant
expansion and chunk-level reranking. The engine is intentionally abstracted so
the same :class:`neoagent.search_tool.SearchTool` can run over either:

* :class:`OfflineCorpusBackend` -- a local JSON corpus indexed with BM25, used
  for trajectory generation and offline development; or
* :class:`WebSearchBackend` -- your live search engine (the configuration used
  to produce the paper's numbers). Implement :meth:`raw_search` / :meth:`fetch`
  against your provider.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .bm25 import BM25


@dataclass
class RawResult:
    url: str
    title: str
    snippet: str
    score: float


class SearchBackend(ABC):
    @abstractmethod
    def raw_search(self, query: str, k: int = 10) -> List[RawResult]:
        """Return up to ``k`` raw results for ``query``."""

    @abstractmethod
    def fetch(self, url: str) -> str:
        """Return the full text of ``url`` (used by BROWSE)."""


class OfflineCorpusBackend(SearchBackend):
    """BM25 over a local JSON corpus of ``{"url","title","text"}`` records."""

    def __init__(self, corpus_path: str | Path):
        self.docs: List[Dict[str, str]] = json.loads(
            Path(corpus_path).read_text(encoding="utf-8")
        )
        self._texts = [f"{d['title']} {d['text']}" for d in self.docs]
        self._bm25 = BM25(self._texts)
        self._by_url = {d["url"]: d for d in self.docs}

    def raw_search(self, query: str, k: int = 10) -> List[RawResult]:
        out = []
        for idx, score in self._bm25.top_k(query, k=k):
            d = self.docs[idx]
            out.append(RawResult(d["url"], d["title"], d["text"][:240], score))
        return out

    def fetch(self, url: str) -> str:
        doc = self._by_url.get(url)
        return doc["text"] if doc else ""


class WebSearchBackend(SearchBackend):
    """Adapter for a live search engine. Fill in with your provider.

    The paper treats the engine as a fixed black box; plug your API here so the
    rest of the pipeline (variant expansion, chunk rerank, summarization) is
    unchanged. Keep ``raw_search`` returning at most ``k`` :class:`RawResult`.
    """

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        self.api_key = api_key
        self.endpoint = endpoint

    def raw_search(self, query: str, k: int = 10) -> List[RawResult]:  # pragma: no cover
        raise NotImplementedError(
            "Connect your search engine here (e.g. Serper/Bing/Google CSE): "
            "issue the query, then return up to k RawResult(url, title, snippet, score)."
        )

    def fetch(self, url: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "Fetch and return the page text for `url` (e.g. via requests + a "
            "readability extractor)."
        )
