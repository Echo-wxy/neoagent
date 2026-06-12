"""Train the character-level variant expander (Section 2.5).

Fine-tunes a TinyBERT-class masked-LM on (canonical, variant) string pairs with
a masked-character reconstruction objective, so it learns to propose plausible
alternative surface forms. Provide a JSONL of pairs:

    {"canonical": "rizzed up", "variant": "r1zz3d up"}

Run:
    python scripts/train_variant_expander.py \
        --pairs data/variant_pairs.jsonl \
        --base huawei-noah/TinyBERT_General_4L_312D \
        --out checkpoints/variant-expander
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="JSONL of {canonical, variant}")
    ap.add_argument("--base", default="huawei-noah/TinyBERT_General_4L_312D")
    ap.add_argument("--out", default="checkpoints/variant-expander")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
    )

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForMaskedLM.from_pretrained(args.base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()

    rows = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]

    class PairDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            # Train the model to reconstruct canonical characters from the
            # variant surface form (and vice versa), at the character level.
            r = rows[i]
            text = f"{r['variant']} = {r['canonical']}"
            enc = tok(text, truncation=True, max_length=64)
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=args.mask_prob)
    loader = DataLoader(PairDataset(), batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total += out.loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  mean_loss={total / max(1, len(loader)):.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved variant expander to {args.out}")


if __name__ == "__main__":
    main()
