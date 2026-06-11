# NeoAgent: An Optimization-Driven Framework for Variant-Aware Retrieval-Augmented Question Answering

> Code and data accompanying the paper *"NeoAgent: An optimization-driven framework for
> variant-aware retrieval-augmented question answering"* (PeerJ Computer Science, AI Application).
> Archived on Zenodo: **DOI 10.5281/zenodo.20308619** — https://doi.org/10.5281/zenodo.20308619

---

## 1. Description

NeoAgent is an autonomous information-seeking agent for retrieval-augmented question answering
over **network neologisms** (slang, abbreviations, leetspeak, homophonic rewrites, and other
fast-drifting surface forms). It is built on the ReAct paradigm and is designed around one
principle: *the lexical regime under which the agent is trained must match the regime it meets at
inference time.*

The repository provides everything needed to reproduce the study:

1. a **variant-aware data-synthesis pipeline** (neologism evolution tree → variant generation and
   constraint optimization → variant-aware QA generation with verification);
2. a **customized search/browse environment** with on-the-fly variant expansion;
3. the **supervised fine-tuning (SFT) recipe** that trains NeoAgent on 15k demonstration
   trajectories; and
4. the **evaluation scripts** used to produce every reported number.

## 2. Dataset Information

**Synthesized training data (produced by this repository).**
15,000 demonstration trajectories generated with the teacher model OpenAI o3 (API snapshot
`o3-2025-04-16`) running in the variant-expanded search environment. Seed neologisms were
collected from public social-media streams and dictionary entries; preprocessing consists of
deduplication, removal of entries lacking a reference source, and encoding normalization. **No
personally identifiable information is retained** in the synthesized data.
【请按你的仓库填写：训练数据/种子词表存放的子目录或文件名，及（如适用）下载方式。】

**Third-party evaluation benchmarks (not redistributed here; obtained from official sources).**

| Benchmark | Source / Persistent identifier |
|---|---|
| BrowseComp | https://arxiv.org/abs/2504.12516 (DOI: 10.48550/arXiv.2504.12516) |
| BrowseComp-ZH | https://arxiv.org/abs/2504.19314 (DOI: 10.48550/arXiv.2504.19314) |
| Xbench-DeepSearch | https://arxiv.org/abs/2506.13651 (DOI: 10.48550/arXiv.2506.13651); data: https://huggingface.co/datasets/xbench/DeepSearch |
| SimpleQA | https://arxiv.org/abs/2411.04368 (DOI: 10.48550/arXiv.2411.04368) |
| WebWalkerQA | https://arxiv.org/abs/2501.07572 (DOI: 10.48550/arXiv.2501.07572) |

These datasets are governed by their own licenses; please obtain them from the sources above.

## 3. Code Information

The codebase is organized by the pipeline stages described in the paper. 【请按你的仓库实际目录核对/修改下表的路径名。】

| Component | Purpose (paper section) |
|---|---|
| `src/evolution_tree/` | Neologism evolution tree / forest construction (Eq. 1) |
| `src/variant_synthesis/` | Variant generation + constraint optimization, incl. the greedy clue-selection solver (Eqs. 2–3, Algorithm 1) |
| `src/qa_generation/` | QA generation with variant-aware verification (Eq. 4) |
| `src/search_tool/` | Customized Search/Browse environment with on-the-fly variant expansion (Eq. 5) |
| `src/training/` | Supervised fine-tuning recipe (Eq. 6) |
| `src/evaluation/` | Evaluation scripts for the five benchmarks (Avg@4 protocol) |
| `configs/` | Training / inference / tool configuration files |
| `scripts/` | End-to-end entry-point scripts (synthesis → training → evaluation) |

## 4. Requirements

Reproduced on the following environment (from the paper):

- **OS:** Ubuntu 22.04 LTS
- **GPU:** 8 × NVIDIA A100 80 GB (≈ 480 GPU-hours for 3 epochs on the 15k-trajectory dataset)
- **CUDA:** 12.1
- **Python:** 3.10  【请确认你实际使用的版本】
- **Core libraries:** PyTorch 2.1, Hugging Face Transformers, DeepSpeed (ZeRO-3), BF16 mixed precision

Install (example):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

【请按你的仓库填写：提供一个 `requirements.txt`，并尽量固定确切版本，例如
`torch==2.1.*`, `transformers==<版本>`, `deepspeed==<版本>`, 以及 `numpy`, `tqdm`,
`rank_bm25` 或等价 BM25 实现, `sentence-transformers`（Sentence-BERT + cross-encoder reranker）等。
若使用 o3 进行数据合成，还需 `openai` 客户端及环境变量 `OPENAI_API_KEY`。】

## 5. Usage Instructions

```bash
# 0) Clone / unzip the archived deposit, then install requirements (Section 4)

# 1) Build the neologism evolution forest from the seed set
python -m src.evolution_tree.build --seeds <seed_file> --out <forest_dir>

# 2) Variant generation + constraint optimization + variant-aware QA generation
#    (calls OpenAI o3 for candidate generation/verification; set OPENAI_API_KEY)
python -m src.variant_synthesis.run --forest <forest_dir> --out <synth_dir>
python -m src.qa_generation.run --synth <synth_dir> --out <trajectories_dir>

# 3) Supervised fine-tuning (DeepSpeed ZeRO-3, BF16)
deepspeed src/training/train_sft.py --config configs/sft_llama3_8b.yaml \
    --data <trajectories_dir>

# 4) Evaluation on the five benchmarks (Avg@4)
python -m src.evaluation.run --ckpt <checkpoint_dir> --benchmark browsecomp
```

【请按你的仓库实际入口脚本名与参数替换上面的命令；命令应能复现论文中所有报告数值。】

## 6. Methodology

Each instance is a tuple `(x, y, E)` — a question `x`, an answer `y`, and a latent evidence set `E`
indexed by surface forms that may differ from those in `x`. NeoAgent addresses this mismatch in
four stages:

1. **Evolution tree construction** — organize seed neologisms into an evolution forest that models
   morphological and semantic drift, acting as a controllable generator of retrieval confounders.
2. **Variant generation and constraint optimization** — obscure each node's attributes (variant
   obfuscation, numerical/date fuzzification, semantic rephrasing) and select a variant-constrained
   clue set that minimizes the candidate space subject to a no-shortcut (lexical-opacity) constraint,
   solved greedily (Algorithm 1).
3. **Variant-aware QA generation with verification** — generate root-attribute questions and keep
   only instances a strong teacher (o3) solves in ≥ 3 of 5 trials with ≥ 1 search call
   ("difficult but learnable").
4. **Training** — supervised fine-tuning of an 8B base model on 15k o3 trajectories, with the same
   variant-expansion search environment exposed at inference time.

## 7. Citations

If you use this code or data, please cite the paper and the archived deposit.

```bibtex
@article{neoagent2026,
  title   = {NeoAgent: An optimization-driven framework for variant-aware retrieval-augmented question answering},
  author  = {Gao, Ruixi and Wang, Ziqi and Zhao, Zhiyu and Wang, Xinyu and Cheng, Jin},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {【发表后补卷号/期号/页码或文章号与 DOI】}
}

@software{neoagent_code,
  author    = {Gao, Ruixi and Wang, Ziqi and Zhao, Zhiyu and Wang, Xinyu and Cheng, Jin},
  title     = {NeoAgent (source code and data-synthesis pipeline)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20308619},
  url       = {https://doi.org/10.5281/zenodo.20308619}
}
```

Please also cite the third-party base models (Llama-3-8B, Qwen2.5-7B, Mistral-7B) and the five
benchmarks (Section 2) according to their original publications.



*This README documents an exact copy of the code used to perform the study described in the
associated article.*
