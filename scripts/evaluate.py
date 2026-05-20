"""Evaluate NeoAgent on the synthesized QA set and run a small ablation.

Compares the full variant-aware agent against a "w/o variant expansion"
configuration, mirroring the ablation in the paper (Section 3.3.2), and
saves two figures into ``assets/`` for the README.

Run:
    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless backend; no display needed
import matplotlib.pyplot as plt

from neoagent import (
    NeoAgent,
    SearchTool,
    VariantExpander,
    build_forest,
    generate_qa,
    iter_subtree,
    load_seed_nodes,
)

ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "data" / "seed_neologisms.json"
CORPUS_PATH = ROOT / "data" / "web_corpus.json"
ASSETS = ROOT / "assets"

SCHEMAS = ["def", "date", "cat", "sent"]
BLUE = "#2E86AB"
BROWN = "#A0522D"


def build_eval_set(nodes):
    roots = build_forest(nodes)
    qa = []
    for seed_offset in range(3):  # repeat with different stochastic masks
        for i, root in enumerate(roots):
            subtree = iter_subtree(root, max_depth=2)
            qa.append(
                generate_qa(root, subtree, nodes, schema="def",
                            seed=i + 100 * seed_offset)
            )
    return qa


def run_config(nodes, qa_set, use_variant_expansion: bool):
    expander = VariantExpander.from_seed_nodes(nodes)
    tool = SearchTool(
        CORPUS_PATH,
        expander=expander,
        use_variant_expansion=use_variant_expansion,
    )
    agent = NeoAgent(tool, candidates=nodes)

    correct = 0
    total_calls = 0
    for qa in qa_set:
        out = agent.answer(qa.question, gold=qa.answer)
        correct += int(out.correct)
        total_calls += out.n_search_calls
    accuracy = 100.0 * correct / len(qa_set)
    avg_calls = total_calls / len(qa_set)
    return accuracy, avg_calls


def plot_accuracy(full_acc, ablated_acc):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Full NeoAgent", "w/o Variant Expansion"]
    values = [full_acc, ablated_acc]
    bars = ax.bar(labels, values, color=[BLUE, BROWN], width=0.55)
    ax.set_ylabel("Definition accuracy (%)")
    ax.set_title("Variant expansion ablation (offline demo)")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    delta = full_acc - ablated_acc
    ax.annotate(f"+{delta:.1f} pts", xy=(0.5, max(values) + 6),
                xycoords=("axes fraction", "data"), ha="center",
                color=BLUE, fontweight="bold")
    fig.tight_layout()
    out = ASSETS / "ablation_accuracy.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_expansion_example(nodes):
    expander = VariantExpander.from_seed_nodes(nodes)
    query = "3circl3 dr1nk"
    expanded = expander.expand(query)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.axis("off")
    ax.text(0.02, 0.88, "Query", fontsize=11, color=BROWN, fontweight="bold")
    ax.text(0.02, 0.74, f"\u201c{query}\u201d", fontsize=12)
    ax.text(0.02, 0.50, "Variant expansion", fontsize=11, color=BLUE, fontweight="bold")
    for i, var in enumerate(expanded[1:], start=0):
        ax.text(0.06, 0.36 - i * 0.12, f"\u2022 {var}", fontsize=11)
    fig.tight_layout()
    out = ASSETS / "variant_expansion_example.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    nodes = load_seed_nodes(SEED_PATH)
    qa_set = build_eval_set(nodes)

    full_acc, full_calls = run_config(nodes, qa_set, use_variant_expansion=True)
    abl_acc, abl_calls = run_config(nodes, qa_set, use_variant_expansion=False)

    print("=" * 56)
    print(f"Evaluated on {len(qa_set)} synthesized definition questions")
    print("-" * 56)
    print(f"  Full NeoAgent           : {full_acc:5.1f}%  "
          f"({full_calls:.1f} search calls/q)")
    print(f"  w/o Variant Expansion   : {abl_acc:5.1f}%  "
          f"({abl_calls:.1f} search calls/q)")
    print(f"  Improvement             : +{full_acc - abl_acc:.1f} points")
    print("=" * 56)

    p1 = plot_accuracy(full_acc, abl_acc)
    p2 = plot_expansion_example(nodes)
    print(f"Saved figure: {p1.relative_to(ROOT)}")
    print(f"Saved figure: {p2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
