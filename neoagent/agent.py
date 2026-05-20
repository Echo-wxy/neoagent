"""NeoAgent: a transparent ReAct-style solver (Section 2).

This is a lightweight, rule-driven stand-in for the SFT-trained agent. It
demonstrates the *search--reason* loop the paper trains: issue a query,
expand it into variants, retrieve evidence, and score candidate answers
against the recovered snippets. It is fully offline and deterministic so
the repository runs with a single click.

To use a real trained policy, replace :meth:`NeoAgent.answer` with calls to
your fine-tuned model while keeping the same :class:`SearchTool` interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

from .bm25 import tokenize
from .evolution_tree import Node
from .search_tool import SearchResult, SearchTool


@dataclass
class Trace:
    """A single search--reason step, recorded for inspection / SFT export."""

    query: str
    expanded_queries: List[str]
    results: List[SearchResult]
    thought: str


@dataclass
class AgentOutput:
    answer: str
    correct: bool
    n_search_calls: int
    traces: List[Trace] = field(default_factory=list)


class NeoAgent:
    """Variant-aware information-seeking agent over an offline corpus."""

    def __init__(self, search_tool: SearchTool, candidates: Sequence[Node], max_steps: int = 3):
        self.search = search_tool
        self.candidates = list(candidates)
        self.max_steps = max_steps

    def answer(self, question: str, gold: str | None = None) -> AgentOutput:
        """Run the search--reason loop and return the predicted answer."""
        traces: List[Trace] = []
        evidence: List[tuple[str, float]] = []
        n_calls = 0

        # The synthesized question hides the concept behind a quoted alias
        # (the lexical-drift anchor). Search that alias first so variant
        # expansion has something concrete to resolve; fall back to the full
        # question text.
        alias = self._extract_alias(question)
        query = alias or question

        for step in range(self.max_steps):
            results = self.search.search(query, top_n=3)
            n_calls += 1
            expanded = (
                self.search.expander.expand(query)
                if self.search.use_variant_expansion
                else [query]
            )
            for rank, r in enumerate(results):
                evidence.append((r.snippet, r.score / (1.0 + rank)))

            thought = (
                f"step {step}: query={query!r} -> {len(expanded)} expanded "
                f"queries, {len(results)} snippets."
            )
            traces.append(Trace(query, expanded, results, thought))

            # Reformulate toward the strongest snippet for the next hop while
            # keeping the alias anchored.
            if results:
                query = f"{alias or ''} {results[0].snippet[:80]}".strip()

        prediction = self._score_candidates(evidence)
        correct = bool(gold) and prediction.lower() == gold.lower()
        return AgentOutput(prediction, correct, n_calls, traces)

    @staticmethod
    def _extract_alias(question: str) -> str | None:
        """Pull the obfuscated alias out of the question (text in quotes)."""
        m = re.search(r"'([^']+)'", question)
        return m.group(1) if m else None

    def _score_candidates(self, evidence: Sequence[tuple[str, float]]) -> str:
        """Pick the candidate whose corpus footprint best matches the
        aggregated, score-weighted evidence snippets."""
        weighted_tokens: dict[str, float] = {}
        for text, weight in evidence:
            for tok in tokenize(text):
                weighted_tokens[tok] = weighted_tokens.get(tok, 0.0) + weight

        best_word, best_score = "", -1.0
        for node in self.candidates:
            footprint = set(tokenize(str(node.attributes.get("definition", ""))))
            footprint.update(tokenize(node.word))
            for variant in node.variants:
                footprint.update(tokenize(variant))
            score = sum(weighted_tokens.get(tok, 0.0) for tok in footprint)
            if score > best_score:
                best_score = score
                best_word = node.word
        return best_word
