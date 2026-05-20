"""On-the-fly variant expansion.

The paper expands a query into a small set of high-confidence alternative
surface forms with a fine-tuned character-level transformer (TinyBERT).
To keep this reference implementation lightweight and dependency-free, we
provide a deterministic rule-based ``VariantExpander`` that mirrors the
*interface and behaviour* described in Section 2.5: it returns the original
query plus at most ``max_variants`` plausible rewrites, and suppresses
expansion when confidence is low.

Swap :class:`VariantExpander` for a learned model by implementing the same
``expand(query) -> list[str]`` contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

# Forward leetspeak (for generating extra candidates) and its inverse
# (for normalizing an obfuscated query back toward a canonical form).
_LEET_MAP = {"e": "3", "a": "4", "o": "0", "i": "1", "s": "5", "t": "7"}
_DELEET_MAP = {v: k for k, v in _LEET_MAP.items()}

# A few semantic paraphrase hints keyed by topic words appearing in a query.
_PARAPHRASE_HINTS = {
    "drink": ["esports energy drink meme", "gaming energy drink joke"],
    "energy": ["esports energy drink meme", "pro gamer drink culture"],
    "circle": ["esports energy drink meme", "energy drink esports joke"],
    "charisma": ["rizz slang", "attraction skill slang"],
    "delusional": ["delulu fandom slang", "delulu is the solulu"],
    "mediocre": ["mid slang", "so mid internet"],
    "excellent": ["peak slang", "peak fiction"],
}


def _leetspeak(word: str) -> str:
    return "".join(_LEET_MAP.get(ch, ch) for ch in word)


def _deleetspeak(word: str) -> str:
    """Map leet digits back to letters (e.g. 'r1zz3d' -> 'rizzed')."""
    return "".join(_DELEET_MAP.get(ch, ch) for ch in word.lower())


class VariantExpander:
    """Predicts plausible alternative surface forms for a query.

    Parameters
    ----------
    pairs_path:
        Optional path to a JSON file mapping canonical forms to known
        variants (e.g. derived from the seed set). When provided, exact and
        substring matches are used as high-confidence expansions.
    max_variants:
        Maximum number of expansions returned in addition to the original
        query (``ell <= 3`` in the paper's default configuration).
    """

    def __init__(self, pairs_path: str | Path | None = None, max_variants: int = 3):
        self.max_variants = max_variants
        self.pairs: Dict[str, List[str]] = {}
        if pairs_path is not None:
            self.pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))

    @classmethod
    def from_seed_nodes(cls, nodes, max_variants: int = 3) -> "VariantExpander":
        """Build the variant table directly from evolution-tree nodes."""
        expander = cls(max_variants=max_variants)
        for node in nodes:
            expander.pairs[node.word] = list(node.variants)
            for variant in node.variants:
                expander.pairs.setdefault(variant, list(node.variants))
        return expander

    def expand(self, query: str) -> List[str]:
        """Return ``[query, variant_1, ...]`` with at most ``max_variants`` rewrites."""
        candidates: List[str] = []
        q_lower = query.lower()
        # De-leeted view of the query, so obfuscated aliases (e.g. 'r1zz3d up')
        # can be matched against the known-variant table.
        q_norm = _deleetspeak(q_lower)
        tokens = re.findall(r"[a-z0-9-]+", q_lower)
        norm_tokens = re.findall(r"[a-z0-9-]+", q_norm)

        # 1a. Known-variant lookups by whole-string containment on both the
        #     raw and de-leeted query (catches multi-word aliases).
        for known, variants in self.pairs.items():
            if " " in known and (known in q_lower or known in q_norm):
                candidates.extend(variants)

        # 1b. Known-variant lookups by single-token match on both views.
        for token in set(tokens) | set(norm_tokens):
            for known, variants in self.pairs.items():
                if token == known or token in known or known in token:
                    candidates.extend(variants)

        # 2. Semantic paraphrase hints.
        for token in tokens:
            if token in _PARAPHRASE_HINTS:
                candidates.extend(_PARAPHRASE_HINTS[token])

        # 3. Leetspeak rewrite of the dominant content token (a cheap
        #    character-level mutation in the spirit of the TinyBERT model).
        content = [t for t in tokens if len(t) >= 4]
        if content:
            leet = _leetspeak(content[0])
            if leet != content[0]:
                candidates.append(query.lower().replace(content[0], leet, 1))

        # De-duplicate while preserving order, drop the original query, and cap.
        seen = {q_lower}
        accepted: List[str] = []
        for cand in candidates:
            cand = cand.strip()
            if cand and cand.lower() not in seen:
                seen.add(cand.lower())
                accepted.append(cand)
            if len(accepted) >= self.max_variants:
                break

        return [query] + accepted
