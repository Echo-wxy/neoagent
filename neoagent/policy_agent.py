"""Inference-time policy agent backed by the fine-tuned student model.

At evaluation time the trained Llama-3-8B drives the same ReAct loop and the
same variant-aware SEARCH / BROWSE tools used during trajectory generation. The
student is reached through an OpenAI-compatible endpoint -- e.g. a vLLM server
launched on the SFT checkpoint:

    vllm serve checkpoints/neoagent-llama3-8b --served-model-name neoagent-llama3-8b

then point an :class:`~neoagent.llm.LLMClient` at it via ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .llm import LLMClient
from .search_tool import SearchTool
from .trajectory import Step, generate_trajectory


@dataclass
class AgentAnswer:
    answer: str
    n_search_calls: int
    steps: List[Step]


class PolicyAgent:
    """Runs the trained student over the tools and returns its final answer."""

    def __init__(self, tool: SearchTool, model: LLMClient, max_steps: int = 40):
        self.tool = tool
        self.model = model
        self.max_steps = max_steps

    def answer(self, question: str) -> AgentAnswer:
        traj = generate_trajectory(
            question, gold="", tool=self.tool, teacher=self.model, max_steps=self.max_steps
        )
        return AgentAnswer(traj.final_answer, traj.n_search_calls, traj.steps)
