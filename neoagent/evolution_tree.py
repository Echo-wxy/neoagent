"""Neologism evolution tree / forest construction (Section 2.2).

A :class:`Node` wraps one neologism with its semantic attributes and
observed variants. :func:`build_forest` links related nodes into trees
whose edges represent semantically meaningful drift, yielding the
evolution forest used as a controllable generator of retrieval
confounders (Equation 1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Node:
    """A single neologism node ``v_i = (w_i, ref_i, A_i, V_i)``."""

    word: str
    ref: str
    attributes: Dict[str, object]
    variants: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    children: List["Node"] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Node({self.word!r}, {len(self.children)} children)"


def load_seed_nodes(path: str | Path) -> List[Node]:
    """Load seed neologisms from a JSON file into :class:`Node` objects."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Node(
            word=rec["word"],
            ref=rec.get("ref", ""),
            attributes=rec.get("attributes", {}),
            variants=rec.get("variants", []),
            related=rec.get("related", []),
        )
        for rec in records
    ]


def build_forest(nodes: List[Node], max_fanout: int = 3) -> List[Node]:
    """Link nodes into an evolution forest using their ``related`` edges.

    Each node's ``related`` field plays the role of the candidate-children
    set ``C_i`` from Equation 1 (here pre-extracted rather than produced by a
    live NER--RE operator). We attach up to ``max_fanout`` children per node
    and return the list of root nodes (those never referenced as a child).

    Returns the roots of the forest; every node remains reachable, and the
    full node list is unchanged in place except for populated ``children``.
    """
    by_word: Dict[str, Node] = {n.word: n for n in nodes}
    order = {n.word: i for i, n in enumerate(nodes)}
    child_words: set = set()

    for node in nodes:
        attached = 0
        for rel in node.related:
            if attached >= max_fanout:
                break
            child = by_word.get(rel)
            if child is None or child is node:
                continue
            # Break cycles deterministically: only attach a related node as a
            # child if it appears later in the seed ordering. This turns the
            # symmetric ``related`` graph into a forest of trees.
            if order[child.word] <= order[node.word]:
                continue
            node.children.append(child)
            child_words.add(child.word)
            attached += 1

    roots = [n for n in nodes if n.word not in child_words]
    return roots or nodes


def iter_subtree(root: Node, max_depth: int = 2) -> List[Node]:
    """Breadth-first collection of ``root`` and its descendants."""
    collected: List[Node] = []
    frontier = [(root, 0)]
    seen: set = set()
    while frontier:
        node, depth = frontier.pop(0)
        if node.word in seen or depth > max_depth:
            continue
        seen.add(node.word)
        collected.append(node)
        for child in node.children:
            frontier.append((child, depth + 1))
    return collected
