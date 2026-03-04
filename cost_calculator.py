MODEL_PRICING = {
    "OpenAI": {"input": 0.0005/1000, "output": 0.0015/1000},
    "Groq": {"input": 0.0003/1000, "output": 0.0006/1000},
    "Gemini": {"input": 0.00035/1000, "output": 0.0007/1000}
}


def estimate_cost(model_name, input_tokens, output_tokens):
    if model_name not in MODEL_PRICING:
        return 0

    pricing = MODEL_PRICING[model_name]
    return (input_tokens * pricing["input"]) + \
           (output_tokens * pricing["output"])