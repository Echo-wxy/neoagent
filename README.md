---

## Installation

```bash
git clone <your-repo-url>     # 🔧
cd neoagent
pip install -r requirements.txt
pip install -e .              # optional: import neoagent from anywhere
```

Python ≥ 3.9. The lightweight components (evolution tree, synthesis, BM25) need
only the standard library; `torch`, `transformers`, `deepspeed`, `datasets`,
`sentence-transformers` and `openai` are imported lazily and required only by the
stages that use them.

---

## Reproducing the paper

Run the steps in order. To reproduce the reported numbers you need to supply, as
in the paper: the base weights `meta-llama/Meta-Llama-3-8B`; an OpenAI API key for
the teacher (`o3`, snapshot `o3-2025-04-16`); a live search backend (implement
`WebSearchBackend`); the benchmark datasets; and compute on the order of 8× A100
80GB (~480 GPU-hours for SFT). Reported accuracies are produced by running the
evaluation below — they are not hard-coded anywhere in this repository.

```bash
# 1. Build the neologism evolution forest (Eq. 1)
python scripts/1_build_forest.py --seeds data/seed_neologisms.json

# 2. Synthesize variant-constrained QA (Eq. 2–4, Alg. 1),
#    filtered by teacher variant-aware verification (Sec. 2.4)
OPENAI_API_KEY=... python scripts/2_synthesize_qa.py \
    --seeds data/seed_neologisms.json --out data/qa.jsonl \
    --verify --corpus data/web_corpus.json

# 3. Generate teacher (o3) demonstration trajectories (Sec. 2.4)
OPENAI_API_KEY=... python scripts/3_generate_trajectories.py \
    --qa data/qa.jsonl --out data/trajectories.jsonl --backend web

# 4. Supervised fine-tuning of Llama-3-8B (Eq. 6)
deepspeed --num_gpus 8 -m neoagent.train_sft \
    --model meta-llama/Meta-Llama-3-8B \
    --data data/trajectories.jsonl \
    --output_dir checkpoints/neoagent-llama3-8b \
    --deepspeed configs/deepspeed_zero3.json

# 5. Evaluate under Avg@4 (serve the checkpoint behind an OpenAI-compatible API)
vllm serve checkpoints/neoagent-llama3-8b --served-model-name neoagent-llama3-8b
OPENAI_BASE_URL=http://localhost:8000/v1 python scripts/5_evaluate.py \
    --benchmark browsecomp --data data/benchmarks/browsecomp.jsonl \
    --student neoagent-llama3-8b --backend web
```

The learned variant expander (the paper's character-level TinyBERT) is trained
separately and passed to evaluation via `--variant-model`:

```bash
python scripts/train_variant_expander.py \
    --pairs data/variant_pairs.jsonl --out checkpoints/variant-expander
```

**Ablations.** Pass `--no-variant-expansion` to `scripts/5_evaluate.py` for the
"w/o VarExp" setting; ablate the synthesis operators (variant obfuscation,
numeric fuzzification, semantic rephrasing, evolution tree) via
`configs/synthesis.yaml`.

---

## Training recipe

| Setting | Value |
|---|---|
| Base model | Llama-3-8B |
| Trajectories | 15k (teacher o3, variant-expanded search) |
| Optimizer | AdamW |
| Effective batch size | 128 |
| Epochs | 3 |
| Learning rate | 3e-5, constant with 50 warmup steps |
| Context length | 32k |
| Precision / sharding | BF16 / DeepSpeed ZeRO-3 |
| Reinforcement learning | none (SFT only) |
| Compute | 8× A100 80GB, ~480 GPU-hours |

Encoded in `configs/sft_llama3_8b.yaml`.

---

## Benchmarks

Third-party datasets, used without modifying the released splits. The links
below point to the dataset files themselves (not the accompanying papers), as
required for reproducibility. None of these datasets has an independent
registered DOI (e.g., Zenodo); the arXiv identifiers in the "Paper" column
refer only to the papers, not the data. Download and point
`scripts/5_evaluate.py` at the local files:

| Benchmark | Dataset URL | Format | Paper |
|---|---|---|---|
| BrowseComp | [openai/simple-evals](https://github.com/openai/simple-evals) — [data file](https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv) | CSV | Wei et al., [arXiv:2504.12516](https://doi.org/10.48550/arXiv.2504.12516) |
| BrowseComp-ZH | [PALIN2018/BrowseComp-ZH](https://github.com/PALIN2018/BrowseComp-ZH) (mirror: [HF](https://huggingface.co/datasets/PALIN2018/BrowseComp-ZH)) | XLSX/JSON, encrypted† | Zhou et al., [arXiv:2504.19314](https://doi.org/10.48550/arXiv.2504.19314) |
| Xbench-DeepSearch | [xbench-ai/xbench-evals](https://github.com/xbench-ai/xbench-evals) (mirror: [HF](https://huggingface.co/datasets/xbench/DeepSearch-2510)) | CSV, encrypted† | Chen et al., [arXiv:2506.13651](https://doi.org/10.48550/arXiv.2506.13651) |
| SimpleQA | [openai/simple-evals](https://github.com/openai/simple-evals) — [data file](https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv) | CSV | Wei et al., [arXiv:2411.04368](https://doi.org/10.48550/arXiv.2411.04368) |
| WebWalkerQA | [Alibaba-NLP/WebWalker](https://github.com/Alibaba-NLP/WebWalker) (mirror: [HF](https://huggingface.co/datasets/callanwu/WebWalkerQA)) | JSON | Wu et al., [arXiv:2501.07572](https://doi.org/10.48550/arXiv.2501.07572) |

† Encrypted to prevent search-engine contamination; each official repository
ships a decryption script and canary key.

No personally identifiable information is retained in synthesized training data.

---

## Citation

```bibtex
@article{gao2026neoagent,
  author  = {Gao, Ruixi and Wang, Ziqi and Zhao, Zhiyu and Wang, Xinyu and Cheng, Jin},
  title   = {{NeoAgent}: An Optimization-Driven Framework for Variant-Aware Retrieval-Augmented Question Answering},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Under review}
}
```
Update the volume / pages / DOI once the article is published.

## Data and code availability

An archived snapshot of this repository is deposited on Zenodo:
https://doi.org/10.5281/zenodo.20308619

## Contact

Corresponding author: Jin Cheng (chengjin202605@163.com).

## License

Released under the MIT License. See [LICENSE](LICENSE).
