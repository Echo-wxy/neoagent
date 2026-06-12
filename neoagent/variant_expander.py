"""On-the-fly variant expansion (Section 2.5).

The paper expands a query into a small set (typically <= 3) of high-confidence
alternative surface forms with a fine-tuned **character-level transformer
(TinyBERT)** trained on ~50k (canonical, variant) string pairs with a
masked-character reconstruction objective.

This module provides two interchangeable implementations behind one contract,
``expand(query) -> list[str]`` (the original query first, then accepted variants):

* :class:`LearnedVariantExpander` -- loads a fine-tuned HuggingFace
  character-level model and is the faithful counterpart to the paper. Train it
  with ``scripts/train_variant_expander.py``.
* :class:`RuleVariantExpander` -- a dependency-free deterministic fallback used
  for smoke tests and when no checkpoint is available. It is **not** the paper's
  model; it only mirrors the interface and the "few high-confidence variants,
  suppress low-confidence ones" behaviour.

Use :func:`build_expander` to pick the learned model when a checkpoint path is
given, else the rule-based fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol

# --------------------------------------------------------------------------
# Common interface
# --------------------------------------------------------------------------


class VariantExpander(Protocol):
    def expand(self, query: str) -> List[str]:
        ...


# --------------------------------------------------------------------------
# Learned expander (faithful to the paper)
# --------------------------------------------------------------------------
class LearnedVariantExpander:
    """Character-level model that proposes alternative surface forms.

    Loads a fine-tuned encoder (TinyBERT-class) and, for each query, samples a
    small set of high-confidence rewrites by reconstructing masked characters.
    Candidates below ``min_confidence`` are suppressed to protect precision, and
    at most ``max_variants`` are returned (the paper's ``ell <= 3`` default).

    Parameters
    ----------
    model_path:
        Path or HF id of the fine-tuned character-level checkpoint.
    max_variants, min_confidence:
        Acceptance controls matching Section 2.5.
    device:
        Torch device string.
    """

    def __init__(
        self,
        model_path: str,
        max_variants: int = 3,
        min_confidence: float = 0.5,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.max_variants = max_variants
        self.min_confidence = min_confidence
        self.device = device
        self._tok = None
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_path)
        self._model.to(self.device).eval()

    def expand(self, query: str) -> List[str]:
        """Return ``[query, variant_1, ...]`` with confident rewrites only."""
        self._ensure_model()
        import torch

        # Character-level masked reconstruction: mask one substitutable position
        # at a time and read off high-probability character substitutions, then
        # assemble the resulting candidate surface forms.
        candidates: Dict[str, float] = {}
        chars = list(query)
        for i, ch in enumerate(chars):
            if not ch.isalnum():
                continue
            masked = "".join(chars[:i] + [self._tok.mask_token or "[MASK]"] + chars[i + 1:])
            enc = self._tok(masked, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            mask_positions = (enc["input_ids"] == self._tok.mask_token_id).nonzero(as_tuple=True)
            if mask_positions[0].numel() == 0:
                continue
            pos = mask_positions[1][0]
            probs = torch.softmax(logits[0, pos], dim=-1)
            top_prob, top_id = probs.max(dim=-1)
            if top_prob.item() < self.min_confidence:
                continue
            sub = self._tok.convert_ids_to_tokens(int(top_id)).lstrip("##")
            if len(sub) == 1 and sub.isalnum() and sub != ch:
                cand = "".join(chars[:i] + [sub] + chars[i + 1:])
                candidates[cand] = max(candidates.get(cand, 0.0), top_prob.item())

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        accepted = [c for c, _ in ranked[: self.max_variants]]
        return [query] + accepted


# --------------------------------------------------------------------------
# Rule-based fallback (NOT the paper's model -- smoke-test stand-in)
# --------------------------------------------------------------------------
_LEET_MAP = {"e": "3", "a": "4", "o": "0", "i": "1", "s": "5", "t": "7"}
_DELEET_MAP = {v: k for k, v in _LEET_MAP.items()}


def _leetspeak(word: str) -> str:
    return "".join(_LEET_MAP.get(ch, ch) for ch in word)


def _deleetspeak(word: str) -> str:
    return "".join(_DELEET_MAP.get(ch, ch) for ch in word.lower())


class RuleVariantExpander:
    """Deterministic, dependency-free expander for smoke tests / fallback."""

    def __init__(self, pairs_path: Optional[str | Path] = None, max_variants: int = 3):
        self.max_variants = max_variants
        self.pairs: Dict[str, List[str]] = {}
        if pairs_path is not None:
            self.pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))

    @classmethod
    def from_seed_nodes(cls, nodes, max_variants: int = 3) -> "RuleVariantExpander":
        exp = cls(max_variants=max_variants)
        for node in nodes:
            exp.pairs[node.word] = list(node.variants)
            for variant in node.variants:
                exp.pairs.setdefault(variant, list(node.variants))
        return exp

    def expand(self, query: str) -> List[str]:
        candidates: List[str] = []
        q_lower = query.lower()
        q_norm = _deleetspeak(q_lower)
        tokens = re.findall(r"[a-z0-9-]+", q_lower)
        norm_tokens = re.findall(r"[a-z0-9-]+", q_norm)

        for known, variants in self.pairs.items():
            if " " in known and (known in q_lower or known in q_norm):
                candidates.extend(variants)
        for token in set(tokens) | set(norm_tokens):
            for known, variants in self.pairs.items():
                if token == known or token in known or known in token:
                    candidates.extend(variants)

        content = [t for t in tokens if len(t) >= 4]
        if content:
            leet = _leetspeak(content[0])
            if leet != content[0]:
                candidates.append(q_lower.replace(content[0], leet, 1))

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


def build_expander(
    model_path: Optional[str] = None,
    seed_nodes=None,
    max_variants: int = 3,
    **kwargs,
) -> VariantExpander:
    """Return the learned expander if ``model_path`` is given, else the rule fallback."""
    if model_path:
        return LearnedVariantExpander(model_path, max_variants=max_variants, **kwargs)
    if seed_nodes is not None:
        return RuleVariantExpander.from_seed_nodes(seed_nodes, max_variants=max_variants)
    return RuleVariantExpander(max_variants=max_variants)
