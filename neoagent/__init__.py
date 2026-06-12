"""NeoAgent: an optimization-driven framework for variant-aware
retrieval-augmented question answering.

Reference implementation of the full pipeline described in the paper:

  evolution_tree     neologism evolution forest                  (Eq. 1)
  synthesis          variant-constrained data synthesis          (Eq. 2-4, Alg. 1)
  variant_expander   on-the-fly variant expansion (learned/rule) (Sec. 2.5)
  retrieval          dense filter + cross-encoder rerank         (Sec. 2.5)
  search_backend     pluggable raw retrieval (offline / web)
  search_tool        variant-aware SEARCH / BROWSE               (Eq. 5)
  llm                teacher (o3) / student chat client
  trajectory         ReAct demonstration-trajectory generation   (Sec. 2.4)
  verification       variant-aware acceptance of synthetic QA    (Sec. 2.4)
  data, train_sft    SFT of Llama-3-8B on trajectories           (Eq. 6)
  policy_agent       inference with the fine-tuned student
  eval               benchmark evaluation under Avg@4

Heavy dependencies (torch, transformers, sentence-transformers, openai) are
imported lazily inside the functions that need them, so importing this package
is cheap and the lightweight components (evolution tree, synthesis, BM25) run on
the standard library alone.
"""

from .bm25 import BM25, tokenize
from .data import DataCollatorForTrajectory, TrajectoryDataset, load_chat_records
from .evolution_tree import Node, build_forest, iter_subtree, load_seed_nodes
from .llm import LLMClient, Message
from .policy_agent import AgentAnswer, PolicyAgent
from .retrieval import CrossEncoderReranker, DenseRetriever
from .search_backend import (
    OfflineCorpusBackend,
    RawResult,
    SearchBackend,
    WebSearchBackend,
)
from .search_tool import SearchResult, SearchTool
from .synthesis import (
    QAInstance,
    generate_qa,
    greedy_clue_selection,
    obscure_attributes,
)
from .trajectory import Step, Trajectory, dump_trajectories, generate_trajectory
from .variant_expander import (
    LearnedVariantExpander,
    RuleVariantExpander,
    build_expander,
)
from .verification import VerificationResult, is_correct, verify_instance

__version__ = "1.0.0"

__all__ = [
    "BM25", "tokenize",
    "Node", "build_forest", "iter_subtree", "load_seed_nodes",
    "QAInstance", "generate_qa", "greedy_clue_selection", "obscure_attributes",
    "build_expander", "LearnedVariantExpander", "RuleVariantExpander",
    "DenseRetriever", "CrossEncoderReranker",
    "SearchBackend", "OfflineCorpusBackend", "WebSearchBackend", "RawResult",
    "SearchTool", "SearchResult",
    "LLMClient", "Message",
    "Trajectory", "Step", "generate_trajectory", "dump_trajectories",
    "verify_instance", "is_correct", "VerificationResult",
    "TrajectoryDataset", "DataCollatorForTrajectory", "load_chat_records",
    "PolicyAgent", "AgentAnswer",
]
