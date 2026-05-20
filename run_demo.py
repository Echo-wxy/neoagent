"""Click-to-run entry point: synthesize -> evaluate -> plot.

Runs the full lightweight pipeline end to end and prints a worked example
of the variant-aware search--reason loop.

Run:
    python run_demo.py
"""

from __future__ import annotations

from pathlib import Path

from neoagent import (
    NeoAgent,
    SearchTool,
    VariantExpander,
    build_forest,
    generate_qa,
    iter_subtree,
    load_seed_nodes,
)

ROOT = Path(__file__).resolve().parent
SEED_PATH = ROOT / "data" / "seed_neologisms.json"
CORPUS_PATH = ROOT / "data" / "web_corpus.json"


def main() -> None:
    print("\n# 1. Build neologism evolution forest -------------------------")
    nodes = load_seed_nodes(SEED_PATH)
    roots = build_forest(nodes)
    print(f"   {len(nodes)} seed nodes -> {len(roots)} forest roots")
    for root in roots[:3]:
        kids = ", ".join(c.word for c in root.children) or "(leaf)"
        print(f"   - {root.word}: children = {kids}")

    print("\n# 2. Synthesize a variant-constrained question ----------------")
    root = roots[0]
    subtree = iter_subtree(root, max_depth=2)
    qa = generate_qa(root, subtree, nodes, schema="def", seed=0)
    print(f"   target word : {qa.target_word}")
    print(f"   gold answer : {qa.answer}")
    print(f"   question    : {qa.question[:160]}...")

    print("\n# 3. Variant-aware search--reason loop -------------------------")
    expander = VariantExpander.from_seed_nodes(nodes)
    tool = SearchTool(CORPUS_PATH, expander=expander)
    agent = NeoAgent(tool, candidates=nodes)
    out = agent.answer(qa.question, gold=qa.answer)

    for trace in out.traces:
        print(f"   {trace.thought}")
        if trace.results:
            print(f"      top hit: {trace.results[0].title}")
    print(f"\n   predicted answer : {out.answer}")
    print(f"   correct          : {out.correct}")
    print(f"   search calls     : {out.n_search_calls}")

    print("\nDone. Run 'python scripts/evaluate.py' to produce the figures.\n")


if __name__ == "__main__":
    main()
