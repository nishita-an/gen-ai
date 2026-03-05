import json
import logging
# Import the strategy and client you already configured
from llm_clients import hf_client, hf_retry_strategy

def evaluate_response(prompt, response, criteria):
    json_schema_instruction = """
    You must respond ONLY with a valid JSON object matching this exact schema:
    {
        "accuracy": <integer 1-10>,
        "brevity": <integer 1-10>,
        "tone": <integer 1-10>,
        "overall": <integer 1-10>
    }
    """

    evaluation_prompt = f"""
    You are a strict evaluator.
    Criteria: {criteria}
    Original Prompt: {prompt}
    Response to Evaluate: {response}
    Score from 1-10 on Accuracy, Brevity, and Tone based on the criteria.
    {json_schema_instruction}
    """

    # Wrap the call in the retry strategy you already defined
    @hf_retry_strategy
    def _get_eval():
        return hf_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output strict JSON."},
                {"role": "user", "content": evaluation_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500
        )

    try:
        result = _get_eval()
        return result.choices[0].message.content
    except Exception as e:
        logging.error(f"Evaluation failed for DeepSeek-V3: {e}")
        # Return a dummy JSON so the rest of your dashboard doesn't crash
        return json.dumps({"accuracy": 0, "brevity": 0, "tone": 0, "overall": 0, "error": str(e)})