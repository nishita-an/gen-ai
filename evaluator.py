import json
import logging
import re
# Import the updated Groq components from your new llm_clients module
from llm_clients import groq_client, groq_retry_strategy, call_qwen

def _extract_json(text):
    """Robustly extracts JSON from a string, handling markdown blocks or leading text."""
    try:
        # Find anything between curly braces
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception:
        return None

def evaluate_response(prompt, response, criteria):
    # The Rubric helps the LLM anchor its scores
    rubric = """
    1-3: Poor. Factual errors, ignores constraints, or rude tone.
    4-6: Fair. Correct but lacks detail, minor formatting issues, or slightly off-tone.
    7-9: Good. Professional, accurate, and follows all instructions.
    10: Perfect. Exceptional insight, perfectly formatted, and highly concise.
    """

    evaluation_prompt = f"""
    You are an expert AI Quality Auditor. Your task is to grade an LLM response based on a user prompt.

    [USER PROMPT]
    {prompt}

    [MODEL RESPONSE TO GRADE]
    {response}

    [CRITERIA]
    {criteria}

    [SCORING RUBRIC]
    {rubric}

    [INSTRUCTIONS]
    1. Analyze the response qualitatively in 2 sentences.
    2. Provide scores (1-10) for accuracy, brevity, and tone.
    3. Output the result ONLY in this JSON format:
    {{
        "reasoning": "your 2-sentence analysis here",
        "accuracy": int,
        "brevity": int,
        "tone": int,
        "overall": int
    }}
    """

    @groq_retry_strategy
    def _get_eval_primary():
        # Groq supports the JSON object response format for Llama 3 models
        return groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a strict, objective JSON evaluator. Output only valid JSON."},
                {"role": "user", "content": evaluation_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

    # --- EXECUTION WITH FALLBACK ---
    try:
        # Try Llama-3.3-70b via Groq first
        result = _get_eval_primary()
        content = result.choices[0].message.content
        parsed = _extract_json(content)
        if parsed: return parsed
    except Exception as e:
        logging.warning(f"Llama 3.3 Eval failed: {e}. Falling back to Qwen-3-32B.")
        
    try:
        # Fallback to Qwen (via your updated llm_clients logic)
        fallback = call_qwen(evaluation_prompt)
        parsed = _extract_json(fallback["response"])
        if parsed: return parsed
    except Exception as e:
        logging.error(f"Fallback Eval also failed: {e}")

    # Ultimate Safety Return
    return {"accuracy": 0, "brevity": 0, "tone": 0, "overall": 0, "reasoning": "Evaluation failed."}