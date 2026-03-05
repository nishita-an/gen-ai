

MODEL_PRICING = {
    "DeepSeek-V3": {"input": 0.00014/1000, "output": 0.00028/1000}, # HF/DeepSeek Rates
    "Qwen-72B": {"input": 0.0003/1000, "output": 0.0006/1000},
    "Groq (Llama 3.3)": {"input": 0.00059/1000, "output": 0.00079/1000},
}


def estimate_cost(model_name, input_tokens, output_tokens):
    if model_name not in MODEL_PRICING:
        return 0

    pricing = MODEL_PRICING[model_name]
    return (input_tokens * pricing["input"]) + \
           (output_tokens * pricing["output"])