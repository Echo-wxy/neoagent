"""Variant-constrained data synthesis (Section 2.3 / Algorithm 1).

Implements the three stochastic attribute transformations of Equation 2
(variant obfuscation, numeric fuzzification, semantic rephrasing) and the
greedy solver for the variant-constrained clue-selection problem of
Equation 3 / Algorithm 1.

The greedy procedure selects clues that shrink the candidate-neologism
space (the ``eta`` estimator) while never letting a known variant become
directly recoverable by the top-K retrieval probe (the ``rho_K`` constraint
enforcing lexical opacity).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .bm25 import BM25
from .evolution_tree import Node


# --------------------------------------------------------------------------
# Equation 2: three stochastic attribute-set transformations
# --------------------------------------------------------------------------
_LEET_OBFUSCATE = {"e": "3", "a": "4", "o": "0", "i": "1", "s": "5"}


def _obfuscate(form: str, rng: random.Random) -> str:
    """Apply a light leetspeak mutation so the alias no longer lexically
    overlaps the canonical surface form found in the corpus. This is the
    severe surface-form corruption the agent must undo via variant expansion."""
    out = []
    for ch in form:
        repl = _LEET_OBFUSCATE.get(ch.lower())
        # Mutate roughly half of the substitutable characters.
        out.append(repl if (repl and rng.random() < 0.6) else ch)
    return "".join(out)


def f_variant(node: Node, rng: random.Random) -> Dict[str, str]:
    """Variant obfuscation: surface the concept through an *obfuscated*
    variant (leetspeak/homophone-style noise), never the canonical form.
    This is the lexical drift that variant expansion recovers at retrieval."""
    aliases = [v for v in node.variants if v.lower() != node.word.lower()]
    base = rng.choice(aliases) if aliases else (
        node.variants[0] if node.variants else node.word
    )
    masked = _obfuscate(base, rng)
    clues = {"masked_term": f"a slang form rendered as '{masked}'"}
    if "definition" in node.attributes:
        clues["definition"] = str(node.attributes["definition"])
    return clues


def f_static(clues: Dict[str, str], node: Node, rng: random.Random) -> Dict[str, str]:
    """Numeric / date fuzzification: turn exact counts and years into ranges."""
    out = dict(clues)
    year = node.attributes.get("origin_year")
    if isinstance(year, int):
        lo = (year // 2) * 2
        out["era"] = f"around the early {lo}s" if lo < year else f"around {lo}"
    count = node.attributes.get("usage_count")
    if isinstance(count, int):
        bucket = "tens of thousands" if count >= 10000 else "a few thousand"
        out["popularity"] = f"used by {bucket} of posts"
    return out


def f_llm(clues: Dict[str, str], node: Node, rng: random.Random) -> Dict[str, str]:
    """Semantic rephrasing: paraphrase each clue while suppressing the
    surface form. Here we apply light deterministic rewrites; plug in an LLM
    by replacing this function with one honouring the same signature."""
    out = dict(clues)
    sentiment = node.attributes.get("sentiment")
    if sentiment:
        out["tone"] = f"the expression carries a {sentiment} connotation"
    domain = node.attributes.get("domain")
    if domain:
        out["context"] = f"it circulates mainly in {domain} communities"
    return out


def obscure_attributes(node: Node, rng: random.Random) -> Dict[str, str]:
    """Apply the Equation-2 pipeline f_variant -> f_static -> f_llm."""
    clues = f_variant(node, rng)
    clues = f_static(clues, node, rng)
    clues = f_llm(clues, node, rng)
    return clues


# --------------------------------------------------------------------------
# Equation 3 / Algorithm 1: greedy variant-constrained clue selection
# --------------------------------------------------------------------------
def make_eta(all_nodes: Sequence[Node]) -> Callable[[Dict[str, str]], List[str]]:
    """Candidate-neologism estimator ``eta``.

    Returns all seed words whose definition is still consistent with the
    given clue subset (a cheap keyword-overlap proxy for the paper's small
    LM + entity-linker ensemble). Smaller output ==> more discriminative.
    """
    defs = {n.word: str(n.attributes.get("definition", "")).lower() for n in all_nodes}

    def eta(clue_subset: Dict[str, str]) -> List[str]:
        text = " ".join(clue_subset.values()).lower()
        keywords = {w for w in text.split() if len(w) >= 5}
        if not keywords:
            return [n.word for n in all_nodes]
        consistent = []
        for word, definition in defs.items():
            if any(kw in definition for kw in keywords):
                consistent.append(word)
        return consistent or [n.word for n in all_nodes]

    return eta


def make_rho_k(surface_forms: Sequence[str], k: int = 100) -> Callable[[Dict[str, str]], List[str]]:
    """Top-K retrieval probe ``rho_K`` over an index of surface forms."""
    bm25 = BM25(list(surface_forms))

    def rho_k(clue_subset: Dict[str, str]) -> List[str]:
        query = " ".join(clue_subset.values())
        hits = bm25.top_k(query, k=k)
        return [surface_forms[i] for i, _ in hits]

    return rho_k


def greedy_clue_selection(
    obscured: Dict[str, str],
    variants: Sequence[str],
    eta: Callable[[Dict[str, str]], List[str]],
    rho_k: Callable[[Dict[str, str]], List[str]],
) -> Dict[str, str]:
    """Algorithm 1: greedily grow a clue set that is maximally discriminative
    subject to the lexical-opacity constraint ``V_i ∩ rho_K(K) = ∅``."""
    variant_set = {v.lower() for v in variants}
    selected: Dict[str, str] = {}
    remaining = dict(obscured)

    while remaining:
        # arg min over candidate clues of |eta(K ∪ {a})|
        best_key = None
        best_size = None
        for key, value in remaining.items():
            trial = dict(selected)
            trial[key] = value
            size = len(eta(trial))
            if best_size is None or size < best_size:
                best_size = size
                best_key = key

        trial = dict(selected)
        trial[best_key] = remaining[best_key]

        probe = {form.lower() for form in rho_k(trial)}
        if variant_set & probe:
            # Adding this clue would violate lexical opacity -> stop.
            break
        selected[best_key] = remaining.pop(best_key)

    # Always keep at least one clue so the instance stays answerable.
    if not selected and obscured:
        first = next(iter(obscured))
        selected[first] = obscured[first]
    return selected


# --------------------------------------------------------------------------
# Equation 4: QA generation from a sub-tree
# --------------------------------------------------------------------------
@dataclass
class QAInstance:
    question: str
    answer: str
    target_word: str
    schema: str
    clues: Dict[str, str]


_SCHEMA_TEMPLATES = {
    "def": "Based on the following clues, what online expression is being described? {clues}",
    "date": "Around when did the expression described below first appear? {clues}",
    "cat": "Which community or domain does the expression below belong to? {clues}",
    "sent": "What overall sentiment does the expression described below carry? {clues}",
}

_SCHEMA_ANSWER_KEY = {
    "def": "word",
    "date": "origin_year",
    "cat": "domain",
    "sent": "sentiment",
}


def generate_qa(
    root: Node,
    subtree_nodes: Sequence[Node],
    all_nodes: Sequence[Node],
    schema: str = "def",
    seed: int = 0,
) -> QAInstance:
    """Synthesize one QA instance (Equation 4) anchored on ``root``.

    The question surfaces the root concept only through an obfuscated
    *variant* (lexical drift) plus short, paraphrased hints drawn from the
    root and its neighbours. The canonical surface form never appears, so a
    literal query cannot match the evidence page directly -- recovering it
    requires variant expansion at retrieval time.
    """
    rng = random.Random(seed)
    surface_forms = [n.word for n in all_nodes] + [
        v for n in all_nodes for v in n.variants
    ]
    eta = make_eta(all_nodes)
    rho_k = make_rho_k(surface_forms, k=20)

    # Root is surfaced through an obfuscated variant (kept as the lexical
    # anchor) plus a paraphrased, variant-constrained definition hint.
    root_obscured = obscure_attributes(root, random.Random(seed + 1))
    masked_term = root_obscured.pop("masked_term", None)

    # Apply the opacity-constrained greedy selection to the *remaining*
    # (non-anchor) clues so the definition cannot be trivially looked up.
    root_clues = greedy_clue_selection(root_obscured, root.variants, eta, rho_k)

    # Neighbours contribute short categorical hints only (no full
    # definitions), so the root stays the latent anchor rather than a sibling.
    neighbour_hints: Dict[str, str] = {}
    for i, node in enumerate(subtree_nodes):
        if node.word == root.word:
            continue
        domain = node.attributes.get("domain")
        if domain:
            neighbour_hints[f"{node.word}.context"] = (
                f"it is discussed alongside {domain} slang"
            )

    merged_clues: Dict[str, str] = {}
    if masked_term is not None:
        merged_clues[f"{root.word}.masked_term"] = masked_term
    for key, value in root_clues.items():
        merged_clues[f"{root.word}.{key}"] = value
    merged_clues.update(neighbour_hints)

    clue_text = " ".join(merged_clues.values())
    question = _SCHEMA_TEMPLATES[schema].format(clues=clue_text)

    if schema == "def":
        answer = root.word
    else:
        key = _SCHEMA_ANSWER_KEY[schema]
        answer = str(root.attributes.get(key, root.word))

    return QAInstance(
        question=question,
        answer=answer,
        target_word=root.word,
        schema=schema,
        clues=merged_clues,
    )
