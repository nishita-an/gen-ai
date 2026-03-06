# Groq On-Demand API Pricing (Calculated per 1k tokens)
# Sourced from Groq's current pricing for per-million tokens
MODEL_PRICING = {
    "GPT-OSS-120B": {"input": 0.00015/1000, "output": 0.00060/1000}, # $0.15 in / $0.60 out per 1M
    "Llama-3.3-70B": {"input": 0.00059/1000, "output": 0.00079/1000}, # $0.59 in / $0.79 out per 1M
    "Qwen-3-32B": {"input": 0.00029/1000, "output": 0.00059/1000}, # $0.29 in / $0.59 out per 1M
}

def estimate_cost(model_name, input_tokens, output_tokens):
    """Calculates the estimated cost of an API call based on token usage."""
    if model_name not in MODEL_PRICING:
        return 0.0

    pricing = MODEL_PRICING[model_name]
    
    # Calculate total cost for the request
    total_cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
    
    return total_cost