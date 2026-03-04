import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")