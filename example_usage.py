"""
example_usage.py
─────────────────
Demonstrates the MemoryAgent running a 15-turn study conversation
without the FastAPI server layer.

Requirements:
  export GROQ_API_KEY=<your key>
  pip install -r requirements.txt
  cd memory_agent
  python example_usage.py
"""

import os
import sys
import logging
from pathlib import Path
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv  # <--- Add this

# Load variables from .env into the system environment
load_dotenv()
# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.WARNING)   # suppress debug noise in demo

from services.memory_agent import MemoryAgent

DEMO_CONVERSATION = [
    "Hi! I'm studying machine learning and I want to master transformer architectures this month.",
    "Can you explain what the attention mechanism does in a transformer?",
    "What's the difference between self-attention and cross-attention?",
    "I'm working with PyTorch. Can you show me the core of a scaled dot-product attention implementation?",
    "What's the purpose of the scaling factor 1/sqrt(d_k)?",
    "My project is building a text summarisation model for scientific papers. Which transformer variant should I use?",
    "What are the key hyperparameters I need to tune for an encoder-decoder model?",
    "I keep getting NaN losses during training — what should I check first?",
    "I have a constraint: my model must run inference on a laptop CPU in under 2 seconds per document.",
    "Given that constraint, should I use distillation or pruning to reduce model size?",
    "Let's switch topics — can you explain backpropagation through time (BPTT)?",
    "Why does BPTT suffer from vanishing gradients for long sequences?",
    "How does an LSTM solve the vanishing gradient problem?",
    "Circling back to my summarisation project — given everything we've discussed, what architecture would you recommend?",
    "What should my training pipeline look like, step by step?",
]


def run_demo():
    print("=" * 65)
    print("  Persistent Memory Study Assistant — Demo")
    print("=" * 65)

    
    if not os.getenv("GROQ_API_KEY"):        
        print("\n⚠  GROQ_API_KEY not set.")
        print("   Running in MOCK mode — responses will be placeholder text.\n")
        _patch_llm_for_demo()

    agent = MemoryAgent(session_id="demo-session")

    for i, message in enumerate(DEMO_CONVERSATION, 1):
        print(f"\n{'─' * 65}")
        print(f"[Turn {i:02d}] USER: {message}")
        print()

        result = agent.chat(message)

        print(f"  ASSISTANT: {result.reply[:400]}{'...' if len(result.reply) > 400 else ''}")
        print()
        print(f"  ⟨ tokens≈{result.token_estimate} | "
              f"latency={result.latency_ms:.0f}ms | "
              f"compressed={'✓' if result.compression_triggered else '✗'} ⟩")

    print(f"\n{'=' * 65}")
    print("  Final Memory Status")
    print("=" * 65)
    status = agent.status()
    for k, v in status.items():
        print(f"  {k:<25} {v}")

    print(f"\n  User Profile:\n")
    print(agent.user_profile.format_for_prompt())
    print()


def _patch_llm_for_demo():
    """Replace the LLM client with a mock so the demo works without an API key."""
    from unittest.mock import MagicMock, patch
    import services.memory_agent as ma_module
    import llm.grok_client as llm_module

    mock_llm = MagicMock()
    mock_llm.chat.return_value = (
        "Great question! [MOCK RESPONSE — set GROQ_API_KEY for real answers] "
        "This is a placeholder response that demonstrates the memory system is "
        "working correctly. The actual LLM would provide a detailed, context-aware "
        "answer here, drawing on your profile, past summaries, and semantic memory."
    )
    mock_llm.count_tokens.side_effect = llm_module.count_tokens_simple

    # Patch at the module level so MemoryAgent picks it up
    import services.summarizer as sum_module
    import services.fact_extractor as fe_module

    sum_module.get_llm_client = lambda: mock_llm
    fe_module.get_llm_client = lambda: mock_llm
    ma_module.get_llm_client = lambda: mock_llm


if __name__ == "__main__":
    run_demo()
