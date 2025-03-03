import os
import json
import openai
import numpy as np
import faiss
from flask import Flask, request, jsonify
from rulebook_processor import EmbeddingProcessor
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

processor = EmbeddingProcessor()

@app.route("/chat", methods=["POST"])
def chat():
    """API endpoint to chat with either fine-tuned model or embedding-based retrieval."""
    data = request.json
    game_name = data.get("game")
    user_query = data.get("query")
    use_fine_tuned = data.get("fine_tuned", False)  # If True, use fine-tuned model

    if not game_name or not user_query:
        return jsonify({"error": "Missing 'game' or 'query' in request."}), 400

    if use_fine_tuned:
        response = ask_fine_tuned(user_query)
    else:
        response = ask_with_embeddings(game_name, user_query)
    
    return jsonify(response)

def ask_fine_tuned(user_query):
    """Queries the fine-tuned GPT-4o Mini model."""
    model_path = "../data/fine_tuned_model.json"
    
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            model_info = json.load(f)
        fine_tuned_model = model_info.get("model_id", "gpt-4o-mini")  # Uses actual fine-tuned model ID
    else:
        fine_tuned_model = "gpt-4o-mini"

    chat_response = client.chat.completions.create(
        model=fine_tuned_model,
        messages=[
            {"role": "system", "content": "You are an expert on board game rules."},
            {"role": "user", "content": user_query}
        ]
    )
    return {"response": chat_response.choices[0].message.content}

def ask_with_embeddings(game_name, user_query):
    """Uses FAISS embedding search to retrieve relevant rule text before asking GPT-4o Mini."""
    found, relevant_text = processor.search_relevant_text(user_query, game_name)
    
    if found:
        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are an expert on board game rules for {game_name}."},
                {"role": "user", "content": f"Relevant rules: {relevant_text}\n\nQuestion: {user_query}"}
            ]
        )
        return {"response": chat_response.choices[0].message.content}
    else:
        return {"error": "No relevant rules found for this game."}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)