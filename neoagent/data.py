"""SFT dataset and collator for trajectory learning (Equation 6).

Each trajectory is a chat record ``{"messages": [...]}`` produced by
:mod:`neoagent.trajectory`. We tokenize it with the base model's chat template
and supervise only the assistant turns (the interleaved reasoning, tool calls
and final answer), masking system / user / tool tokens out of the loss. This is
the standard SFT instantiation of

    L_SFT(theta) = - sum_{(pi, x) in D} sum_t log p_theta(u_t | u_<t, x)

restricted to the agent's own tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

IGNORE_INDEX = -100


def load_chat_records(path: str | Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TrajectoryDataset:
    """Tokenizes chat-format trajectories into (input_ids, labels).

    Parameters
    ----------
    records:
        List of ``{"messages": [...]}`` dicts.
    tokenizer:
        A HuggingFace tokenizer with a chat template (Llama-3).
    max_length:
        Hard cap on sequence length (paper: 32k).
    """

    def __init__(self, records: List[dict], tokenizer, max_length: int = 32768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [self._encode(r["messages"]) for r in records]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]

    def _encode(self, messages: List[dict]) -> Dict[str, List[int]]:
        input_ids: List[int] = []
        labels: List[int] = []
        # Build the sequence turn by turn so we can mask non-assistant tokens.
        for i, msg in enumerate(messages):
            segment = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_generation_prompt=False
            )
            # Avoid duplicating the BOS the template may prepend per call.
            if i > 0 and self.tokenizer.bos_token_id is not None and segment and segment[0] == self.tokenizer.bos_token_id:
                segment = segment[1:]
            input_ids.extend(segment)
            if msg["role"] == "assistant":
                labels.extend(segment)
            else:
                labels.extend([IGNORE_INDEX] * len(segment))

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollatorForTrajectory:
    """Pads a batch of variable-length (input_ids, labels) on the right."""

    tokenizer: object

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, "torch.Tensor"]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids, labels, attn = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
