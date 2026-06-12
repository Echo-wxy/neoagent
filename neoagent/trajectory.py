"""Demonstration-trajectory generation (Section 2.4 / Training).

For each accepted training instance ``(q, y)`` we roll out the **teacher**
(OpenAI o3) in a ReAct loop over the variant-aware SEARCH / BROWSE tools and
record the interleaved reasoning, tool calls, observations and final answer.
The serialized trajectory is the supervision signal for the student SFT
(Equation 6).

Nothing here is faked: each step is a real teacher call and a real tool
execution. With the offline backend you get fully reproducible local
trajectories; point the backend at your web search to reproduce the paper's
setup.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .llm import LLMClient, Message
from .search_tool import SearchTool

REACT_SYSTEM = """You are a meticulous web-research agent. Solve the question by \
interleaving reasoning with tool calls. The corpus indexes evidence under \
canonical surface forms, while the question may use obfuscated variants, so \
search aggressively and reconcile evidence across hops.

Respond on each turn in exactly one of these formats:
  Thought: <your reasoning>
  Action: SEARCH <query>
or
  Thought: <your reasoning>
  Action: BROWSE <url> <what you are looking for>
or
  Thought: <your reasoning>
  Action: FINISH <final answer>
"""

_ACTION_RE = re.compile(r"Action:\s*(SEARCH|BROWSE|FINISH)\s*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass
class Step:
    thought: str
    action: str          # "SEARCH" | "BROWSE" | "FINISH"
    argument: str
    observation: str = ""


@dataclass
class Trajectory:
    question: str
    gold: str
    steps: List[Step] = field(default_factory=list)
    final_answer: str = ""
    n_search_calls: int = 0

    def to_chat_record(self) -> dict:
        """Serialize to an OpenAI-style messages record for SFT."""
        messages = [{"role": "system", "content": REACT_SYSTEM},
                    {"role": "user", "content": self.question}]
        for step in self.steps:
            assistant = f"Thought: {step.thought}\nAction: {step.action} {step.argument}".strip()
            messages.append({"role": "assistant", "content": assistant})
            if step.action.upper() != "FINISH":
                messages.append({"role": "tool", "content": step.observation})
        return {"question": self.question, "gold": self.gold, "messages": messages}


def _render_observation(result_block: str) -> str:
    return f"Observation:\n{result_block}"


def generate_trajectory(
    question: str,
    gold: str,
    tool: SearchTool,
    teacher: LLMClient,
    max_steps: int = 40,
) -> Trajectory:
    """Roll out the teacher over the tools and return the recorded trajectory."""
    traj = Trajectory(question=question, gold=gold)
    history: List[Message] = [
        Message("system", REACT_SYSTEM),
        Message("user", question),
    ]

    for _ in range(max_steps):
        reply = teacher.chat(history)
        history.append(Message("assistant", reply))

        m = _ACTION_RE.search(reply)
        thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", reply, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        if not m:
            # Malformed turn -> nudge once and continue.
            obs = "Observation: please answer with a Thought and an Action line."
            history.append(Message("tool", obs))
            continue

        action = m.group(1).upper()
        argument = m.group(2).strip()

        if action == "FINISH":
            traj.steps.append(Step(thought, action, argument))
            traj.final_answer = argument
            break

        if action == "SEARCH":
            traj.n_search_calls += 1
            results = tool.search(argument)
            block = "\n".join(f"[{r.title}] {r.snippet} (url: {r.url})" for r in results) or "(no results)"
        else:  # BROWSE
            parts = argument.split(None, 1)
            url = parts[0] if parts else ""
            focus = parts[1] if len(parts) > 1 else question
            block = tool.browse(url, focus) or "(page unavailable)"

        observation = _render_observation(block)
        traj.steps.append(Step(thought, action, argument, observation))
        history.append(Message("tool", observation))

    return traj


def dump_trajectories(trajectories: List[Trajectory], path: str) -> None:
    """Write trajectories as JSONL chat records for SFT."""
    with open(path, "w", encoding="utf-8") as fh:
        for traj in trajectories:
            fh.write(json.dumps(traj.to_chat_record(), ensure_ascii=False) + "\n")
