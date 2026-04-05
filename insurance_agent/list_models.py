import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Initialize the Google GenAI client
client = genai.Client()

print("Currently Available Embedding Models:")
for model in client.models.list():
    # Filter the list to only show embedding models
    if "embedding" in model.name:
        print(f"- {model.name} (Display Name: {model.display_name})")