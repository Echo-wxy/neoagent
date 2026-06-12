"""Step 2 -- synthesize variant-constrained QA instances (Eq. 2-4, Alg. 1).

Generates questions whose target concept is hidden behind obfuscated variants,
optionally filtering them with variant-aware verification by the teacher model
(Section 2.4). Verification requires OPENAI_API_KEY and a search backend.

    # offline, no verification (fast smoke test)
    python scripts/2_synthesize_qa.py --seeds data/seed_neologisms.json \
        --out data/qa.jsonl

    # with teacher verification over the offline corpus
    python scripts/2_synthesize_qa.py --seeds data/seed_neologisms.json \
        --out data/qa.jsonl --verify --corpus data/web_corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neoagent import (
    build_expander,
    build_forest,
    generate_qa,
    iter_subtree,
    load_seed_nodes,
)

SCHEMAS = ["def", "date", "cat", "sent"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/seed_neologisms.json")
    ap.add_argument("--out", default="data/qa.jsonl")
    ap.add_argument("--repeats", type=int, default=3, help="stochastic masks per root")
    ap.add_argument("--verify", action="store_true", help="filter with teacher verification")
    ap.add_argument("--corpus", default=None, help="offline corpus for verification")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--min-correct", type=int, default=3)
    args = ap.parse_args()

    nodes = load_seed_nodes(args.seeds)
    roots = build_forest(nodes)

    instances = []
    for r in range(args.repeats):
        for i, root in enumerate(roots):
            subtree = iter_subtree(root, max_depth=2)
            for schema in SCHEMAS:
                instances.append(generate_qa(root, subtree, nodes, schema=schema,
                                             seed=i + 1000 * r + 7 * SCHEMAS.index(schema)))
    print(f"Synthesized {len(instances)} candidate QA instances.")

    if args.verify:
        from neoagent import LLMClient, SearchTool, verify_instance
        from neoagent.search_backend import OfflineCorpusBackend

        if not args.corpus:
            raise SystemExit("--verify requires --corpus (a JSON corpus to search).")
        backend = OfflineCorpusBackend(args.corpus)
        tool = SearchTool(backend=backend, expander=build_expander(seed_nodes=nodes))
        teacher = LLMClient()  # OpenAI o3; needs OPENAI_API_KEY
        kept = []
        for qa in instances:
            res = verify_instance(qa.question, qa.answer, tool, teacher,
                                  trials=args.trials, min_correct=args.min_correct)
            if res.accepted:
                kept.append(qa)
        print(f"Verification kept {len(kept)} / {len(instances)} instances.")
        instances = kept

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for qa in instances:
            fh.write(json.dumps({
                "question": qa.question, "answer": qa.answer,
                "target_word": qa.target_word, "schema": qa.schema,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(instances)} instances to {args.out}")


if __name__ == "__main__":
    main()
