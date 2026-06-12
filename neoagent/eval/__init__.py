"""Benchmark evaluation under the Avg@4 protocol."""

from .benchmarks import BENCHMARK_SOURCES, load_benchmark, load_jsonl
from .scorer import LLMJudge, avg_at_k, normalized_match

__all__ = [
    "BENCHMARK_SOURCES",
    "load_benchmark",
    "load_jsonl",
    "LLMJudge",
    "avg_at_k",
    "normalized_match",
]
