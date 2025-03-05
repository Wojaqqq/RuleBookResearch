import os
import json
import openai
import faiss
import numpy as np
from flask import Flask, request, render_template
from dotenv import load_dotenv
from config import Config
from pathlib import Path

load_dotenv()
app = Flask(__name__)
config = Config.get_instance()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EmbeddingSearch:
    EMBEDDING_METADATA_FILE = config.DATA_DIR / "embedding_metadata.json"

    def __init__(self):
        self.vector_store = self._load_vector_store()
        self.metadata = self._load_metadata()
        print(f"Checking for embeddings at: {config.DATA_DIR}")
        print(f"Vector store file exists: {config.VECTOR_STORE_FILE.exists()}")
        print(f"Embedding metadata exists: {EmbeddingSearch.EMBEDDING_METADATA_FILE.exists()}")

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

        if not self.metadata or self.vector_store.ntotal == 0:
            return False, ""

        query_embedding = self._get_embedding(query).astype("float32").reshape(1, -1)
        _, indices = self.vector_store.search(query_embedding, 5)

        for idx in indices[0]:
            entry = self.metadata[idx]
            if entry["game"].strip().lower() == mapped_game_name:
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
        selected_models = request.form.getlist("models")

        responses = {}

        if "gpt4o" in selected_models:
            responses["gpt4o"] = ask_regular_gpt(query, game)["response"]

        if "fine_tuned" in selected_models:
            responses["fine_tuned"] = ask_fine_tuned(query, game)["response"]

        if "embedding" in selected_models:
            found, embedding_response = ask_with_embeddings(game, query)
            if not found:
                embedding_response = f"No relevant rules found for {game}."
            responses["embedding"] = embedding_response

        return render_template("index.html", games=config.GAMES, selected_game=game, query=query,
                               responses=responses, selected_models=selected_models)

    return render_template("index.html", games=config.GAMES, responses={}, selected_models=[])


def ask_fine_tuned(user_query, game_name):
    if config.FINE_TUNED_MODEL_PATH.exists():
        with open(config.FINE_TUNED_MODEL_PATH, "r") as f:
            model = json.load(f).get("model_id", "gpt-4o-mini")
    else:
        model = "gpt-4o-mini"

    messages = [
        {"role": "system", "content": f"You are an expert on {game_name}. Explain the rules clearly and in simple terms to help players understand."},
        {"role": "user", "content": user_query}
    ]
    return ask_gpt_with_messages(messages, model)


def ask_regular_gpt(user_query, game_name):
    messages = [
        {"role": "system", "content": f"You are a helpful board game assistant. We are talking about game {game_name}"},
        {"role": "user", "content": user_query}
    ]
    return ask_gpt_with_messages(messages, "gpt-4o-mini")


def ask_with_embeddings(game_name, user_query):
    search_engine = EmbeddingSearch()
    found, relevant_text = search_engine.search_relevant_text(user_query, game_name)

    if not found:
        return False, ""

    prompt = f"Here is a rule fragment from {game_name}:\n\n{relevant_text}\n\nExplain this rule in clear and simple terms."
    messages = [
        {"role": "system", "content": f"You are an expert on {game_name}. Explain the rules clearly and in simple terms to help players understand."},
        {"role": "user", "content": prompt}
    ]
    explanation = ask_gpt_with_messages(messages, "gpt-4o-mini")["response"]

    return True, explanation


def ask_gpt_with_messages(messages, model):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return {"response": response.choices[0].message.content}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
