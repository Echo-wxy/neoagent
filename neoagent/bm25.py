"""Pure-Python BM25 ranking function.

Implements Okapi BM25 (Robertson & Zaragoza, 2009) with no external
dependencies. This backs the ``rho_K`` retrieval probe used during data
synthesis and the ``BM25_k`` chunk retriever used at inference time
(Equation 5 in the paper).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used throughout the package."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed corpus of documents.

    Parameters
    ----------
    corpus:
        A sequence of raw document strings.
    k1, b:
        Standard BM25 hyper-parameters.
    """

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens: List[List[str]] = [tokenize(doc) for doc in corpus]
        self.doc_count = len(self.corpus_tokens)
        self.doc_len = [len(toks) for toks in self.corpus_tokens]
        self.avg_len = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0

        # Document frequency for every term.
        df: Counter = Counter()
        for toks in self.corpus_tokens:
            for term in set(toks):
                df[term] += 1
        self.df = df

        # Inverse document frequency (with the BM25 +1 smoothing).
        self.idf = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        self.term_freqs: List[Counter] = [Counter(toks) for toks in self.corpus_tokens]

    def score(self, query: str, index: int) -> float:
        """BM25 score of ``query`` against document ``index``."""
        score = 0.0
        freqs = self.term_freqs[index]
        dl = self.doc_len[index]
        for term in tokenize(query):
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0.0)
            tf = freqs[term]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
            score += idf * (tf * (self.k1 + 1)) / (denom or 1)
        return score

    def top_k(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """Return ``[(doc_index, score), ...]`` for the top-``k`` documents."""
        scored = [(i, self.score(query, i)) for i in range(self.doc_count)]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
