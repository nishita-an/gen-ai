import time
import os
import logging
from huggingface_hub import InferenceClient
from groq import Groq
from config import HUGGINGFACE_API_TOKEN
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup basic logging to see retry attempts in your console
logging.basicConfig(level=logging.INFO)

# Initialize HF client with a significantly longer timeout (120s)
hf_client = InferenceClient(token=HUGGINGFACE_API_TOKEN, timeout=120)

# Groq API Setup
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        from config import GROQ_API_TOKEN
        groq_api_key = GROQ_API_TOKEN
    except ImportError:
        raise ValueError("GROQ_API_KEY not found in environment or config.py")

groq_client = Groq(api_key=groq_api_key)

# Retry decorator: Retries 3 times, waiting longer each time (4s, 8s, 10s max)
# only triggers on general Exceptions (like your 504 error)
hf_retry_strategy = retry(
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

@hf_retry_strategy
def _safe_hf_call(model_id, prompt):
    """Internal helper to handle the actual API call with retries."""
    return hf_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )

def call_deepseek(prompt):
    start_time = time.time()
    model_name = "DeepSeek-V3"
    try:
        response = _safe_hf_call("deepseek-ai/DeepSeek-V3-0324", prompt)
        latency = time.time() - start_time
        return {
            "model": model_name,
            "response": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency": latency
        }
    except Exception as e:
        return format_error(model_name, e)

def call_qwen(prompt):
    start_time = time.time()
    model_name = "Qwen-72B"
    try:
        response = _safe_hf_call("Qwen/Qwen2.5-72B-Instruct", prompt)
        latency = time.time() - start_time
        return {
            "model": model_name,
            "response": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency": latency
        }
    except Exception as e:
        return format_error(model_name, e)

def call_groq(prompt):
    """Call Groq model and return response with metrics."""
    start_time = time.time()
    model_name = "Groq (Mixtral)"
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        latency = time.time() - start_time
        return {
            "model": model_name,
            "response": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency": latency
        }
    except Exception as e:
        return format_error(model_name, e)