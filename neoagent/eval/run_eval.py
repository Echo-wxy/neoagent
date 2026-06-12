"""Evaluate a trained NeoAgent checkpoint on a benchmark (Avg@4).

Wires the served student model + variant-aware tools into a
:class:`~neoagent.policy_agent.PolicyAgent`, runs the benchmark under the Avg@4
protocol, and prints the measured accuracy. The number printed is whatever the
model actually scores on your data and serving stack -- it is computed here, not
assumed.

Example:
    OPENAI_BASE_URL=http://localhost:8000/v1 \
    python -m neoagent.eval.run_eval \
        --benchmark browsecomp --data data/benchmarks/browsecomp.jsonl \
        --student neoagent-llama3-8b \
        --variant-model checkpoints/variant-expander \
        --backend web
"""

from __future__ import annotations

import argparse

from ..llm import LLMClient
from ..policy_agent import PolicyAgent
from ..search_backend import OfflineCorpusBackend, WebSearchBackend
from ..search_tool import SearchTool
from ..variant_expander import build_expander
from .benchmarks import load_benchmark
from .scorer import LLMJudge, avg_at_k, normalized_match


def parse_args():
    p = argparse.ArgumentParser(description="NeoAgent benchmark evaluation")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--data", required=True, help="local benchmark file")
    p.add_argument("--student", default="neoagent-llama3-8b",
                   help="served model name (reached via OPENAI_BASE_URL)")
    p.add_argument("--variant-model", default=None,
                   help="path to the learned variant expander; rule-based if omitted")
    p.add_argument("--backend", choices=["web", "offline"], default="web")
    p.add_argument("--corpus", default=None, help="JSON corpus for the offline backend")
    p.add_argument("--k", type=int, default=4, help="Avg@k runs")
    p.add_argument("--judge", action="store_true", help="use an LLM grader")
    p.add_argument("--no-variant-expansion", action="store_true",
                   help='ablation: disable variant expansion ("w/o VarExp")')
    p.add_argument("--question-field", default="question")
    p.add_argument("--answer-field", default="answer")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    backend = (
        OfflineCorpusBackend(args.corpus)
        if args.backend == "offline"
        else WebSearchBackend()
    )
    expander = build_expander(model_path=args.variant_model)
    tool = SearchTool(
        backend=backend,
        expander=expander,
        use_variant_expansion=not args.no_variant_expansion,
    )

    student = LLMClient(model=args.student, temperature=0.7)
    agent = PolicyAgent(tool, student)

    items = load_benchmark(
        args.benchmark, args.data,
        question_field=args.question_field, answer_field=args.answer_field,
    )

    grader = LLMJudge(LLMClient(model=args.student, temperature=0.0)) if args.judge else normalized_match
    result = avg_at_k(items, lambda q: agent.answer(q).answer, grader=grader, k=args.k)

    print("=" * 60)
    print(f"Benchmark : {args.benchmark}  (n={result['n']})")
    print(f"Variant expansion: {'OFF (ablation)' if args.no_variant_expansion else 'ON'}")
    print(f"Avg@{result['k']} accuracy : {result['avg_at_k']:.1f}%")
    print(f"Per-run   : {[round(x, 1) for x in result['runs']]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
