"""Answer-level scoring and the Avg@4 protocol (Assessment Metrics).

We report answer-level accuracy: a prediction is correct if it matches the
reference under the benchmark's scoring protocol. For NeoAgent (and the
InfoAgent baseline) the paper reports **Avg@4**, the mean accuracy over four
independent runs, to reduce the variance of a stochastic agent trajectory.

Two graders are provided:

* :func:`normalized_match` -- deterministic containment/normalization match.
* :class:`LLMJudge`        -- a model grader for benchmarks whose official
  protocol uses one (e.g. SimpleQA / BrowseComp graded answers). Supply an
  :class:`~neoagent.llm.LLMClient`.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from ..llm import LLMClient, Message


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .\"'")


def normalized_match(prediction: str, gold: str) -> bool:
    p, g = _normalize(prediction), _normalize(gold)
    return bool(g) and (g in p or p in g)


class LLMJudge:
    """Model-graded correctness, for benchmarks whose protocol requires it."""

    def __init__(self, client: LLMClient):
        self.client = client

    def __call__(self, prediction: str, gold: str) -> bool:
        msg = [
            Message("system", "You grade answers. Reply with exactly 'CORRECT' or 'INCORRECT'."),
            Message("user", f"Reference answer: {gold}\nModel answer: {prediction}\n"
                            f"Is the model answer correct?"),
        ]
        verdict = self.client.chat(msg, temperature=0.0, max_tokens=4)
        return "CORRECT" in verdict.upper()


def avg_at_k(
    items: List[dict],
    answer_fn: Callable[[str], str],
    grader: Callable[[str, str], bool] = normalized_match,
    k: int = 4,
) -> Dict[str, float]:
    """Run ``answer_fn`` ``k`` times per item and average accuracy over runs.

    Parameters
    ----------
    items:
        ``[{"question","answer"}, ...]``.
    answer_fn:
        Maps a question to the agent's answer string (one stochastic run).
    grader:
        Correctness function ``(prediction, gold) -> bool``.
    k:
        Number of independent runs (paper: 4).
    """
    per_run_acc: List[float] = []
    for _ in range(k):
        correct = 0
        for item in items:
            pred = answer_fn(item["question"])
            correct += int(grader(pred, item["answer"]))
        per_run_acc.append(100.0 * correct / max(1, len(items)))
    mean = sum(per_run_acc) / len(per_run_acc)
    return {"avg_at_k": mean, "runs": per_run_acc, "k": k, "n": len(items)}
