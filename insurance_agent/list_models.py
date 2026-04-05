import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

print("Currently Available Embedding Models:")
for model in client.models.list():
    if "embedding" in model.name:
        print(f"- {model.name} (Display Name: {model.display_name})")