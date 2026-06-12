"""Step 4 -- supervised fine-tuning of Llama-3-8B (Eq. 6).

For multi-GPU training, launch the module directly with DeepSpeed:

    deepspeed --num_gpus 8 -m neoagent.train_sft \
        --model meta-llama/Meta-Llama-3-8B \
        --data data/trajectories.jsonl \
        --output_dir checkpoints/neoagent-llama3-8b \
        --deepspeed configs/deepspeed_zero3.json

This wrapper forwards to the same entry point for single-process runs / debugging.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neoagent.train_sft import main

if __name__ == "__main__":
    main()
