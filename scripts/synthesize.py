"""Generate variant-constrained QA instances from the seed set.

Run:
    python -m scripts.synthesize          # from the repo root
    python scripts/synthesize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neoagent import build_forest, generate_qa, iter_subtree, load_seed_nodes

ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "data" / "seed_neologisms.json"
OUT_PATH = ROOT / "data" / "synthesized_qa.json"

SCHEMAS = ["def", "date", "cat", "sent"]


def main() -> None:
    nodes = load_seed_nodes(SEED_PATH)
    roots = build_forest(nodes)

    instances = []
    for i, root in enumerate(roots):
        subtree = iter_subtree(root, max_depth=2)
        schema = SCHEMAS[i % len(SCHEMAS)]
        qa = generate_qa(root, subtree, nodes, schema=schema, seed=i)
        instances.append(
            {
                "question": qa.question,
                "answer": qa.answer,
                "target_word": qa.target_word,
                "schema": qa.schema,
                "n_clues": len(qa.clues),
            }
        )

    OUT_PATH.write_text(json.dumps(instances, indent=2), encoding="utf-8")

    print(f"Synthesized {len(instances)} QA instances from {len(nodes)} seed nodes "
          f"({len(roots)} forest roots).")
    print(f"Written to {OUT_PATH.relative_to(ROOT)}\n")
    for inst in instances[:3]:
        print(f"[{inst['schema']}] target={inst['target_word']!r} "
              f"answer={inst['answer']!r} clues={inst['n_clues']}")
        print(f"    Q: {inst['question'][:140]}...\n")


if __name__ == "__main__":
    main()
