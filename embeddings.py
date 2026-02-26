import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def embed(text: str):
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text}
    )
    if not response.ok:
        raise Exception(f"Failed to generate embeddings: {response.text}")
    return response.json()
