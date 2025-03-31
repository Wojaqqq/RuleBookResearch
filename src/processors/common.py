import os
import openai
from pathlib import Path
from dotenv import load_dotenv
from config import Config

load_dotenv()
config = Config.get_instance()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not found!")
    print("Please set this in your .env file or environment.")
    client = None
else:
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(f"ERROR initializing OpenAI client: {e}")
        client = None


def parse_game_name(game_name):
    parts = game_name.split("_", 1)
    result = {
        "base_name": parts[0],
        "is_expansion": len(parts) > 1,
        "expansion_name": parts[1] if len(parts) > 1 else None,
    }
    return result


def chunk_text(text, chunk_size=800):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
