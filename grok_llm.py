import requests  # type: ignore
from config import GROQ_API_KEY, GROQ_MODEL

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_groq(prompt: str) -> str:
    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )

    if response.status_code != 200:
        print(f"[ERROR] Groq API returned {response.status_code}")
        print(f"[ERROR] Response: {response.text}")
    
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]
