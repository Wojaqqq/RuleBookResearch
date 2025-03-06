import os
import json
import openai
import faiss
import numpy as np
import time
from flask import Flask, request, render_template
from dotenv import load_dotenv
from config import Config
from pathlib import Path
from database import save_result, init_db
from datetime import datetime

load_dotenv()
app = Flask(__name__)
config = Config.get_instance()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EmbeddingSearch:
    EMBEDDING_METADATA_FILE = config.DATA_DIR / "embedding_metadata.json"

    def __init__(self):
        self.vector_store = self._load_vector_store()
        self.metadata = self._load_metadata()

    def _load_vector_store(self):
        if config.VECTOR_STORE_FILE.exists():
            return faiss.read_index(str(config.VECTOR_STORE_FILE))
        return faiss.IndexFlatL2(1536)

    def _load_metadata(self):
        if self.EMBEDDING_METADATA_FILE.exists():
            with self.EMBEDDING_METADATA_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def search_relevant_text(self, query, game_name):
        mapped_game_name = config.EMBEDDING_MAPPING.get(game_name, game_name).strip().lower()
        print(f"[DEBUG] Searching for game in embeddings: {mapped_game_name}")

        if not self.metadata or self.vector_store.ntotal == 0:
            print(f"[DEBUG] No embeddings loaded.")
            return False, "", 0

        query_embedding = self._get_embedding(query).astype("float32").reshape(1, -1)

        # Correct unpacking (distances first, indices second)
        distances, indices = self.vector_store.search(query_embedding, 5)

        print(f"[DEBUG] Embedding search distances: {distances}")

        for idx, distance in zip(indices[0], distances[0]):
            entry = self.metadata[idx]
            if entry["game"].strip().lower() == mapped_game_name:
                print(f"[DEBUG] Found match at distance {distance}")
                return True, entry["text"], len(entry["text"].split())

        print(f"[DEBUG] No match found for {mapped_game_name}")
        return False, "", 0


    def _get_embedding(self, text):
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        return np.array(response.data[0].embedding)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        game = request.form["game"]
        query = request.form["query"]
        chosen_model = request.form.get("chosen_model")

        if chosen_model:
            responses = {
                "gpt4o": request.form["gpt4o_answer"],
                "fine_tuned": request.form["fine_tuned_answer"],
                "embedding": request.form["embedding_answer"]
            }
            token_counts = {
                "gpt4o": int(request.form["gpt4o_tokens"]),
                "fine_tuned": int(request.form["fine_tuned_tokens"]),
                "embedding": int(request.form["embedding_tokens"])
            }
            times = {
                "gpt4o": float(request.form["gpt4o_time"]),
                "fine_tuned": float(request.form["fine_tuned_time"]),
                "embedding": float(request.form["embedding_time"])
            }

            save_result(query, responses, token_counts, times, chosen_model)
            return render_template("index.html", games=config.GAMES, responses={}, selected_models=[])

        selected_models = request.form.getlist("models")

        responses = {}
        token_counts = {}
        times = {}

        if "gpt4o" in selected_models:
            responses["gpt4o"], token_counts["gpt4o"], times["gpt4o"] = ask_regular_gpt(query, game)

        if "fine_tuned" in selected_models:
            responses["fine_tuned"], token_counts["fine_tuned"], times["fine_tuned"] = ask_fine_tuned(query, game)

        if "embedding" in selected_models:
            found, embedding_response, embedding_tokens, embedding_time = ask_with_embeddings(game, query)
            if not found:
                embedding_response = f"No relevant rules found for {game}."
            responses["embedding"] = embedding_response
            token_counts["embedding"] = embedding_tokens
            times["embedding"] = embedding_time

        return render_template("index.html", 
            games=config.GAMES, 
            selected_game=game, 
            query=query,
            responses=responses, 
            selected_models=selected_models,
            tokens=token_counts,
            times=times
        )

    return render_template("index.html", games=config.GAMES, responses={}, selected_models=[])

def ask_fine_tuned(user_query, game_name):
    model_id = getattr(config, 'FINE_TUNED_MODEL_ID', 'gpt-4o-mini')
    messages = [
        {"role": "system", "content": f"You are an expert on {game_name}. Explain the rules clearly and in simple terms."},
        {"role": "user", "content": user_query}
    ]
    return ask_gpt_with_messages(messages, model_id)

def ask_regular_gpt(user_query, game_name):
    messages = [
        {"role": "system", "content": f"You are a helpful board game assistant for {game_name}."},
        {"role": "user", "content": user_query}
    ]
    return ask_gpt_with_messages(messages, "gpt-4o-mini")

def ask_with_embeddings(game_name, user_query):
    search_engine = EmbeddingSearch()

    start_time = time.time()
    found, relevant_text, tokens = search_engine.search_relevant_text(user_query, game_name)
    end_time = time.time()

    embedding_time = end_time - start_time

    if not found:
        return False, "", 0, embedding_time

    prompt = f"Here is a rule fragment from {game_name}:\n\n{relevant_text}\n\nExplain this rule in simple terms."

    return True, prompt, tokens, embedding_time

def ask_gpt_with_messages(messages, model):
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    end_time = time.time()
    tokens_used = response.usage.total_tokens
    answer = response.choices[0].message.content

    return answer, tokens_used, end_time - start_time

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
