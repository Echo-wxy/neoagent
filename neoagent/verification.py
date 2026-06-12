"""Variant-aware verification of synthesized QA instances (Section 2.4).

Before accepting a synthetic example we run the teacher (o3) for several
independent trials against the same search environment used at training time,
and keep only instances that are answered correctly in at least ``min_correct``
of ``trials`` while issuing at least one search call. Instances solvable without
browsing are discarded as too easy; instances that fail too often are discarded
as underdetermined. This keeps the dataset in the "difficult but learnable"
regime.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMClient
from .search_tool import SearchTool
from .trajectory import Trajectory, generate_trajectory


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .\"'")


def is_correct(prediction: str, gold: str) -> bool:
    """Lenient containment match; swap for a benchmark-specific grader if needed."""
    p, g = _normalize(prediction), _normalize(gold)
    return bool(g) and (g in p or p in g)


@dataclass
class VerificationResult:
    accepted: bool
    n_correct: int
    n_trials: int
    used_search: bool
    trajectories: list


def verify_instance(
    question: str,
    gold: str,
    tool: SearchTool,
    teacher: LLMClient,
    trials: int = 5,
    min_correct: int = 3,
) -> VerificationResult:
    """Run ``trials`` independent teacher rollouts and apply the accept rule."""
    n_correct = 0
    any_search = False
    rollouts: list[Trajectory] = []
    for _ in range(trials):
        traj = generate_trajectory(question, gold, tool, teacher)
        rollouts.append(traj)
        any_search = any_search or traj.n_search_calls > 0
        if is_correct(traj.final_answer, gold):
            n_correct += 1

    accepted = (n_correct >= min_correct) and any_search
    return VerificationResult(accepted, n_correct, trials, any_search, rollouts)
