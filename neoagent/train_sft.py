"""Supervised fine-tuning of the student agent (Equation 6, Training).

Faithful to the paper's recipe: base model Llama-3-8B, AdamW, effective batch
size 128, 3 epochs, constant learning rate 3e-5 with 50 warmup steps, context
length capped at 32k, BF16, DeepSpeed ZeRO-3, no RL.

This is real training code: with the base weights, the synthesized trajectory
data and the GPUs described in the paper (8x A100 80GB, ~480 GPU-hours), it
trains the model. It does not and cannot produce the benchmark numbers on its
own -- those come from running :mod:`neoagent.eval` on the resulting checkpoint.

Run (single node, 8 GPUs):
    deepspeed --num_gpus 8 -m neoagent.train_sft \
        --model meta-llama/Meta-Llama-3-8B \
        --data data/trajectories.jsonl \
        --output_dir checkpoints/neoagent-llama3-8b \
        --deepspeed configs/deepspeed_zero3.json
"""

from __future__ import annotations

import argparse
import math


def parse_args():
    p = argparse.ArgumentParser(description="NeoAgent SFT")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--data", required=True, help="JSONL of trajectory chat records")
    p.add_argument("--output_dir", default="checkpoints/neoagent-llama3-8b")
    p.add_argument("--deepspeed", default="configs/deepspeed_zero3.json")
    # Paper defaults.
    p.add_argument("--effective_batch_size", type=int, default=128)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_length", type=int, default=32768)
    p.add_argument("--num_gpus", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from .data import (
        DataCollatorForTrajectory,
        TrajectoryDataset,
        load_chat_records,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    records = load_chat_records(args.data)
    dataset = TrajectoryDataset(records, tokenizer, max_length=args.max_length)
    collator = DataCollatorForTrajectory(tokenizer)

    # Gradient accumulation to reach the effective batch size of 128.
    grad_accum = max(
        1,
        args.effective_batch_size // (args.per_device_batch_size * args.num_gpus),
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        deepspeed=args.deepspeed,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(
        f"[NeoAgent SFT] {len(dataset)} trajectories | "
        f"effective batch {args.per_device_batch_size * args.num_gpus * grad_accum} "
        f"(per-device {args.per_device_batch_size} x {args.num_gpus} gpus x "
        f"{grad_accum} accum) | {args.epochs} epochs | lr {args.lr}"
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[NeoAgent SFT] saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
