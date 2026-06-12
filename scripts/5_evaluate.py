"""Step 5 -- evaluate the trained checkpoint on a benchmark under Avg@4.

Serve the SFT checkpoint behind an OpenAI-compatible endpoint, then:

    OPENAI_BASE_URL=http://localhost:8000/v1 \
    python scripts/5_evaluate.py \
        --benchmark browsecomp --data data/benchmarks/browsecomp.jsonl \
        --student neoagent-llama3-8b --backend web

Add --no-variant-expansion for the "w/o VarExp" ablation, or --judge to grade
with an LLM. This forwards to neoagent.eval.run_eval.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neoagent.eval.run_eval import main

if __name__ == "__main__":
    main()
