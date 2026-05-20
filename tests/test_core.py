"""Smoke and unit tests for the NeoAgent reference implementation.

Run:
    python -m pytest tests/        # if pytest is installed
    python tests/test_core.py      # plain-stdlib fallback runner
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neoagent import (  # noqa: E402
    BM25,
    NeoAgent,
    SearchTool,
    VariantExpander,
    build_forest,
    generate_qa,
    iter_subtree,
    load_seed_nodes,
)

ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "data" / "seed_neologisms.json"
CORPUS_PATH = ROOT / "data" / "web_corpus.json"


def test_bm25_ranks_relevant_doc_first():
    corpus = [
        "the cat sat on the mat",
        "dogs are loyal companions",
        "a cat chased the mouse",
    ]
    bm25 = BM25(corpus)
    top = bm25.top_k("cat", k=2)
    assert top, "expected at least one hit for 'cat'"
    assert top[0][0] in (0, 2), "a cat document should rank first"


def test_forest_has_roots_and_children():
    nodes = load_seed_nodes(SEED_PATH)
    roots = build_forest(nodes)
    assert 0 < len(roots) <= len(nodes)
    # At least one root should have attached children.
    assert any(root.children for root in roots)


def test_variant_expander_resolves_leetspeak():
    nodes = load_seed_nodes(SEED_PATH)
    expander = VariantExpander.from_seed_nodes(nodes)
    expanded = expander.expand("r1zzed up")
    assert expanded[0] == "r1zzed up", "original query must be preserved first"
    assert any("rizz" in cand for cand in expanded[1:]), \
        "expansion should recover the canonical 'rizz' form"


def test_search_tool_returns_results():
    tool = SearchTool(CORPUS_PATH, expander=VariantExpander())
    results = tool.search("energy drink esports", top_n=3)
    assert results, "search should return at least one result"
    assert all(r.url.startswith("http") for r in results)


def test_agent_solves_obfuscated_question():
    nodes = load_seed_nodes(SEED_PATH)
    roots = build_forest(nodes)
    expander = VariantExpander.from_seed_nodes(nodes)
    tool = SearchTool(CORPUS_PATH, expander=expander)
    agent = NeoAgent(tool, candidates=nodes)

    root = roots[0]
    qa = generate_qa(root, iter_subtree(root, max_depth=2), nodes, schema="def", seed=0)
    out = agent.answer(qa.question, gold=qa.answer)
    assert out.n_search_calls >= 1
    assert out.answer == qa.answer, "agent should recover the target with expansion"


def test_variant_expansion_helps():
    """Full agent should not underperform the no-expansion ablation."""
    nodes = load_seed_nodes(SEED_PATH)
    roots = build_forest(nodes)
    expander = VariantExpander.from_seed_nodes(nodes)

    qa_set = [
        generate_qa(r, iter_subtree(r, 2), nodes, schema="def", seed=i)
        for i, r in enumerate(roots)
    ]

    def accuracy(use_expansion: bool) -> float:
        tool = SearchTool(CORPUS_PATH, expander=expander,
                          use_variant_expansion=use_expansion)
        agent = NeoAgent(tool, candidates=nodes)
        correct = sum(agent.answer(qa.question, gold=qa.answer).correct for qa in qa_set)
        return correct / len(qa_set)

    assert accuracy(True) >= accuracy(False)


def _run_all():
    tests = [obj for name, obj in globals().items()
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
