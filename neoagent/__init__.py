"""NeoAgent: variant-aware retrieval-augmented question answering.

Lightweight reference implementation of the core mechanisms described in
"NeoAgent: An Optimization-Driven Framework for Variant-Aware
Retrieval-Augmented Question Answering":

  * neologism evolution tree / forest        (evolution_tree.py)
  * variant-constrained data synthesis        (synthesis.py)
  * variant-aware SEARCH / BROWSE tools        (search_tool.py)
  * a ReAct-style search--reason agent         (agent.py)

Everything runs offline with the Python standard library plus matplotlib
for plotting.
"""

from .agent import AgentOutput, NeoAgent, Trace
from .bm25 import BM25, tokenize
from .evolution_tree import Node, build_forest, iter_subtree, load_seed_nodes
from .search_tool import SearchResult, SearchTool
from .synthesis import QAInstance, generate_qa, greedy_clue_selection
from .variants import VariantExpander

__version__ = "0.1.0"

__all__ = [
    "AgentOutput",
    "NeoAgent",
    "Trace",
    "BM25",
    "tokenize",
    "Node",
    "build_forest",
    "iter_subtree",
    "load_seed_nodes",
    "SearchResult",
    "SearchTool",
    "QAInstance",
    "generate_qa",
    "greedy_clue_selection",
    "VariantExpander",
]
