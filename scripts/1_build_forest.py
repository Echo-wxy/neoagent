"""Step 1 -- build the neologism evolution forest (Eq. 1).

    python scripts/1_build_forest.py --seeds data/seed_neologisms.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neoagent import build_forest, iter_subtree, load_seed_nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/seed_neologisms.json")
    ap.add_argument("--max-fanout", type=int, default=3)
    args = ap.parse_args()

    nodes = load_seed_nodes(args.seeds)
    roots = build_forest(nodes, max_fanout=args.max_fanout)
    print(f"Loaded {len(nodes)} seed neologisms; built {len(roots)} trees.")
    for root in roots:
        subtree = iter_subtree(root, max_depth=2)
        print(f"  root={root.word!r:20s} subtree_size={len(subtree)} "
              f"variants={len(root.variants)}")


if __name__ == "__main__":
    main()
