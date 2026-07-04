<!-- 🔧 Before publishing: (1) set <your-repo-url> below; (2) confirm the MIT license
     (swap for Apache-2.0 + a matching LICENSE file if you prefer); (3) if your final
     file/script names differ from the layout below, adjust the paths to match. -->

# NeoAgent

Official implementation of **NeoAgent: An Optimization-Driven Framework for
Variant-Aware Retrieval-Augmented Question Answering**.

NeoAgent targets the *lexical-novelty* regime, where the same concept surfaces as
an abbreviation, a typo, a homophonic rewrite, leetspeak, or a community-specific
paraphrase, so that a literal query fails to retrieve the evidence before
reasoning even begins. The guiding principle is **lexical-regime alignment**: the
lexical distribution under which the agent is trained must match the one it meets
at inference. The framework instantiates this with three coupled mechanisms — a
neologism evolution forest, a variant-constrained data-synthesis pipeline, and a
variant-expansion search tool — and trains an 8B agent by supervised fine-tuning
on teacher-generated demonstration trajectories.

On five public deep-research benchmarks, NeoAgent attains the best accuracy among
open-source models under 15B parameters while preserving broad short-form
factuality.

---

## What this repository implements

The full pipeline, end to end, mapped to the paper:

| Module | Role | Paper |
|---|---|---|
| `neoagent/evolution_tree.py` | neologism evolution forest | Eq. 1 |
| `neoagent/synthesis.py` | variant-constrained synthesis + greedy clue selection | Eq. 2–4, Alg. 1 |
| `neoagent/variant_expander.py` | on-the-fly variant expansion (learned char-level model; rule fallback) | Sec. 2.5 |
| `neoagent/bm25.py` | Okapi BM25 ranking | Sec. 2.5 |
| `neoagent/retrieval.py` | Sentence-BERT filter + cross-encoder rerank | Sec. 2.5 |
| `neoagent/search_backend.py` | raw retrieval backend (offline corpus / live web) | Sec. 2.5 |
| `neoagent/search_tool.py` | variant-aware SEARCH / BROWSE | Eq. 5 |
| `neoagent/trajectory.py` | ReAct demonstration-trajectory generation (teacher o3) | Sec. 2.4 |
| `neoagent/verification.py` | variant-aware acceptance of synthetic QA | Sec. 2.4 |
| `neoagent/data.py`, `train_sft.py` | SFT of Llama-3-8B on trajectories | Eq. 6 |
| `neoagent/policy_agent.py` | inference with the fine-tuned student | Sec. 2 |
| `neoagent/eval/` | benchmark loaders, Avg@4 scorer, evaluation runner | Results |
