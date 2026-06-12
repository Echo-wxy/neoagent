"""Thin LLM client used in two places of the pipeline.

* The **teacher** (OpenAI o3) generates demonstration trajectories and runs
  variant-aware verification (Sections 2.4 and 2.6 of the paper).
* The **student** (the fine-tuned Llama-3-8B, served behind an
  OpenAI-compatible endpoint such as a vLLM server) is queried at inference
  time by :class:`neoagent.policy_agent.PolicyAgent`.

Both share one tiny interface, :class:`LLMClient`, so the same ReAct loop can
drive either model. Nothing here fabricates output -- every call goes to a
real endpoint and raises if credentials / servers are missing.

Environment
-----------
``OPENAI_API_KEY``     API key for the teacher (and any hosted student).
``OPENAI_BASE_URL``    Optional; point at a vLLM / local server for the student.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant" | "tool"
    content: str


class LLMClient:
    """Minimal chat client over the OpenAI-compatible Chat Completions API.

    Parameters
    ----------
    model:
        Model name. Teacher default mirrors the paper's snapshot
        ``o3-2025-04-16``; for the student, pass the name your serving stack
        exposes (e.g. ``"neoagent-llama3-8b"``).
    base_url:
        Optional OpenAI-compatible endpoint. Leave ``None`` for the official
        OpenAI API (teacher); set it to your vLLM server for the student.
    temperature, max_tokens:
        Decoding controls. Verification uses independent stochastic trials,
        so a non-zero temperature is appropriate there.
    """

    def __init__(
        self,
        model: str = "o3-2025-04-16",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None  # lazily created so importing this module is cheap

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The 'openai' package is required for teacher/student calls. "
                "Install it with `pip install openai`."
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and self._base_url is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before generating "
                "trajectories or running verification with the teacher model."
            )
        self._client = OpenAI(api_key=api_key, base_url=self._base_url, timeout=self.timeout)

    def chat(self, messages: List[Message], **overrides) -> str:
        """Return the assistant message text for ``messages``."""
        self._ensure_client()
        payload = [{"role": m.role, "content": m.content} for m in messages]
        resp = self._client.chat.completions.create(
            model=overrides.get("model", self.model),
            messages=payload,
            temperature=overrides.get("temperature", self.temperature),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content or ""
