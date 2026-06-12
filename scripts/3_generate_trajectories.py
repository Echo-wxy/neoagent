"""Step 3 -- generate demonstration trajectories with the teacher (Sec. 2.4).

Rolls out OpenAI o3 over the variant-aware tools for each accepted QA instance
and writes JSONL chat records for SFT. Requires OPENAI_API_KEY; point the
backend at your web search to reproduce the paper's setup, or use the offline
corpus for local development.

    OPENAI_API_KEY=... python scripts/3_generate_trajectories.py \
        --qa data/qa.jsonl --out data/trajectories.jsonl \
        --backend offline --corpus data/web_corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neoagent import (
    LLMClient,
    SearchTool,
    build_expander,
    dump_trajectories,
    generate_trajectory,
    load_seed_nodes,
)
from neoagent.search_backend import OfflineCorpusBackend, WebSearchBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/qa.jsonl")
    ap.add_argument("--out", default="data/trajectories.jsonl")
    ap.add_argument("--backend", choices=["web", "offline"], default="offline")
    ap.add_argument("--corpus", default="data/web_corpus.json")
    ap.add_argument("--seeds", default="data/seed_neologisms.json")
    ap.add_argument("--variant-model", default=None)
    ap.add_argument("--teacher", default="o3-2025-04-16")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    backend = (
        OfflineCorpusBackend(args.corpus) if args.backend == "offline" else WebSearchBackend()
    )
    expander = build_expander(
        model_path=args.variant_model,
        seed_nodes=load_seed_nodes(args.seeds) if args.variant_model is None else None,
    )
    tool = SearchTool(backend=backend, expander=expander)
    teacher = LLMClient(model=args.teacher)

    qa = [json.loads(l) for l in Path(args.qa).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        qa = qa[: args.limit]

    trajectories = []
    for i, item in enumerate(qa):
        traj = generate_trajectory(item["question"], item["answer"], tool, teacher)
        trajectories.append(traj)
        if (i + 1) % 50 == 0:
            print(f"  generated {i + 1}/{len(qa)} trajectories")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump_trajectories(trajectories, args.out)
    print(f"Wrote {len(trajectories)} trajectories to {args.out}")


if __name__ == "__main__":
    main()
