from grok_llm import call_groq

def evaluate_answer(question: str, answer: str) -> str:
    prompt = f"""
You are a strict technical interviewer.

Question:
{question}

Candidate Answer:
{answer}

Evaluate on:
1. Technical correctness (0-10)
2. Depth (0-10)
3. Clarity (0-10)

Return:
- Individual scores
- Total score out of 30
- Honest feedback
"""

    return call_groq(prompt)
