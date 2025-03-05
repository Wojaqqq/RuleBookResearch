import os
import json
import openai
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from config import Config
from pathlib import Path

load_dotenv()
app = Flask(__name__)
config = Config.get_instance()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GAMES = ["Game 1", "Game 2", "Game 3"]  # Replace with actual game names from your PDFs folder


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
        if not self.metadata or self.vector_store.ntotal == 0:
            return False, ""

        query_embedding = self._get_embedding(query).astype("float32").reshape(1, -1)
        _, indices = self.vector_store.search(query_embedding, 5)

        for idx in indices[0]:
            entry = self.metadata[idx]
            if entry["game"].lower() == game_name.lower():
                return True, entry["text"]

        return False, ""

    def _get_embedding(self, text):
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        return np.array(response.data[0].embedding)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        game = request.form["game"]
        query = request.form["query"]

        fine_tuned_response = ask_fine_tuned(query, game)["response"]
        regular_response = ask_with_embeddings(game, query)["response"]

        return render_template("index.html", games=GAMES, selected_game=game, query=query,
                               fine_tuned_response=fine_tuned_response,
                               regular_response=regular_response)

    return render_template("index.html", games=GAMES)


def ask_fine_tuned(user_query, game_name):
    if config.FINE_TUNED_MODEL_PATH.exists():
        with open(config.FINE_TUNED_MODEL_PATH, "r") as f:
            model = json.load(f).get("model_id", "gpt-4o-mini")
    else:
        model = "gpt-4o-mini"
    return ask_gpt(user_query, model, game_name)


def ask_with_embeddings(game_name, user_query):
    search_engine = EmbeddingSearch()
    found, relevant_text = search_engine.search_relevant_text(user_query, game_name)

    if found:
        prompt = f"Relevant rules: {relevant_text}\n\nQuestion: {user_query}"
        return ask_gpt(prompt, "gpt-4o-mini", game_name)
    else:
        return {"response": f"No relevant rules found for game: {game_name}"}


def ask_gpt(user_query, model, game_name):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are an expert on board game rules for {game_name}."},
            {"role": "user", "content": user_query}
        ]
    )
    return {"response": response.choices[0].message.content}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
