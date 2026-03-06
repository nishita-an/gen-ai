import time
import os
import logging
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup basic logging to see retry attempts and errors in your console
logging.basicConfig(level=logging.INFO)

# Groq API Setup
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        from config import GROQ_API_KEY
        groq_api_key = GROQ_API_KEY
    except ImportError:
        raise ValueError("GROQ_API_KEY not found in environment or config.py")

groq_client = Groq(api_key=groq_api_key)

# Retry decorator: Retries 3 times, waiting longer each time (4s, 8s, 10s max)
# Triggers on general Exceptions (like 504 errors, rate limits, etc.)
groq_retry_strategy = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)

def format_error(model_name, error):
    """Helper to return a consistent error structure."""
    return {
        "model": model_name,
        "response": f"⚠️ Error: {str(error)}",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency": 0
    }

@groq_retry_strategy
def _safe_groq_call(model_id, prompt):
    """Internal helper to handle the actual Groq API call with retries."""
    return groq_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )

def call_model(model_id, model_name, prompt):
    """Generic wrapper to process a Groq model call and track metrics."""
    start_time = time.time()
    try:
        response = _safe_groq_call(model_id, prompt)
        latency = time.time() - start_time
        
        # Safely extract token usage
        usage = getattr(response, 'usage', None)
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return {
            "model": model_name,
            "response": response.choices[0].message.content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency
        }
    except Exception as e:
        logging.error(f"{model_name} API Error: {str(e)}") 
        return format_error(model_name, e)

# --- Specific Model Calling Functions ---

def call_gpt_oss(prompt):
    return call_model("openai/gpt-oss-120b", "GPT-OSS-120B", prompt)

def call_llama(prompt):
    return call_model("llama-3.3-70b-versatile", "Llama-3.3-70B", prompt)

def call_qwen(prompt):
    return call_model("qwen/qwen3-32b", "Qwen-3-32B", prompt)