from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL
import os
from dotenv import load_dotenv

load_dotenv()

def load_embeddings():

    embeddings = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "token": os.getenv("HF_TOKEN")
        }

    )

    return embeddings