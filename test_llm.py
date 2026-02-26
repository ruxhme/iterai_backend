import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN is not set in your .env file!")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token or "dummy-key",
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[
        {
            "role": "user",
            "content": "Describe the process of photosynthesis."
        }
    ],
)

print(completion.choices[0].message.content)
