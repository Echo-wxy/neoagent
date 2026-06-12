"""Loaders for the five public evaluation benchmarks (Experimental Design).

All five are third-party datasets obtained from their official public releases
and used without modifying the released splits. Download them yourself and point
the loaders at the local files; the official sources are:

  BrowseComp      https://doi.org/10.48550/arXiv.2504.12516
  BrowseComp-ZH   https://doi.org/10.48550/arXiv.2504.19314
  Xbench-DS       https://doi.org/10.48550/arXiv.2506.13651
                  (set: https://huggingface.co/datasets/xbench/DeepSearch)
  SimpleQA        https://doi.org/10.48550/arXiv.2411.04368
  WebWalkerQA     https://doi.org/10.48550/arXiv.2501.07572

Each loader returns a list of ``{"id", "question", "answer"}`` dicts. Because the
released formats differ, :func:`load_jsonl` lets you map the field names per
benchmark instead of hard-coding one schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

BENCHMARK_SOURCES: Dict[str, str] = {
    "browsecomp": "https://doi.org/10.48550/arXiv.2504.12516",
    "browsecomp_zh": "https://doi.org/10.48550/arXiv.2504.19314",
    "xbench_ds": "https://doi.org/10.48550/arXiv.2506.13651",
    "simpleqa": "https://doi.org/10.48550/arXiv.2411.04368",
    "webwalkerqa": "https://doi.org/10.48550/arXiv.2501.07572",
}


def load_jsonl(
    path: str | Path,
    question_field: str = "question",
    answer_field: str = "answer",
    id_field: str | None = "id",
) -> List[dict]:
    """Load a JSONL/JSON benchmark file, mapping its fields to the common schema."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    items = []
    for i, row in enumerate(rows):
        items.append(
            {
                "id": str(row.get(id_field, i)) if id_field else str(i),
                "question": row[question_field],
                "answer": str(row[answer_field]),
            }
        )
    return items


def load_benchmark(name: str, path: str | Path, **field_map) -> List[dict]:
    """Load one of the five benchmarks by ``name`` from a local ``path``."""
    if name not in BENCHMARK_SOURCES:
        raise ValueError(
            f"Unknown benchmark '{name}'. Expected one of {list(BENCHMARK_SOURCES)}."
        )
    return load_jsonl(path, **field_map)
