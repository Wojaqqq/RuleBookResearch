import os
import openai
import time
import json
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from config import Config
from database import save_result, init_db, fetch_recent_results
from processors import EmbeddingSearch

load_dotenv()
app = Flask(__name__)
config = Config.get_instance()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ask_gpt(user_query, game_name, model):
    messages = [
        {
            "role": "system",
            "content": f"You are an expert in {game_name}, a popular board game. Using information about the game answer question clearly, concisely, and in simple terms.",
        },
        {"role": "user", "content": user_query},
    ]

    start_time = time.time()

    response = client.chat.completions.create(model=model, messages=messages)

    end_time = time.time()
    tokens_used = response.usage.total_tokens
    answer = response.choices[0].message.content

    return answer, tokens_used, end_time - start_time


def ask_with_embeddings(game_name, user_query):
    if game_name not in config.EMBEDDING_MAPPING.keys():
        return "No embeddings", 0, 0
    search_engine = EmbeddingSearch()

    start_time = time.time()
    found, relevant_text = search_engine.search_relevant_text(user_query, game_name)

    if not found:
        return f"No relevant rules found for {game_name}.", 0, 0

    prompt = f"""
Your task is to extract and explain the most relevant rule from the given rulebook text that directly answers the user's question.
Here is a relevant rule fragment:
{relevant_text}
Now, use information from the rule fragment to answer the following question clearly and concisely:
{user_query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in {game_name}, a popular board game. ",
            },
            {"role": "user", "content": prompt},
        ],
    )
    end_time = time.time()
    embedding_time = end_time - start_time
    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    return answer, tokens_used, embedding_time


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
                "embedding": request.form["embedding_answer"],
            }
            token_counts = {
                "gpt4o": int(request.form["gpt4o_tokens"]),
                "fine_tuned": int(request.form["fine_tuned_tokens"]),
                "embedding": int(request.form["embedding_tokens"]),
            }
            times = {
                "gpt4o": float(request.form["gpt4o_time"]),
                "fine_tuned": float(request.form["fine_tuned_time"]),
                "embedding": float(request.form["embedding_time"]),
            }

            save_result(query, responses, token_counts, times, chosen_model)
            return render_template(
                "index.html", games=config.GAMES, responses={}, selected_models=[]
            )

        selected_models = request.form.getlist("models")

        responses = {}
        token_counts = {}
        times = {}

        if "gpt4o" in selected_models:
            responses["gpt4o"], token_counts["gpt4o"], times["gpt4o"] = ask_gpt(
                query, game, "gpt-4o-mini"
            )

        if "fine_tuned" in selected_models:
            if config.FINE_TUNED_MODEL_PATH.exists():
                with config.FINE_TUNED_MODEL_PATH.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    model_id = metadata.get("model_id")
            else:
                model_id = "gpt-4o-mini"

            responses["fine_tuned"], token_counts["fine_tuned"], times["fine_tuned"] = (
                ask_gpt(query, game, model_id)
            )

        if "embedding" in selected_models:
            embedding_response, embedding_tokens, embedding_time = ask_with_embeddings(
                game, query
            )
            responses["embedding"] = embedding_response
            token_counts["embedding"] = embedding_tokens
            times["embedding"] = embedding_time

        return render_template(
            "index.html",
            games=config.GAMES,
            selected_game=game,
            query=query,
            responses=responses,
            selected_models=selected_models,
            tokens=token_counts,
            times=times,
        )

    return render_template(
        "index.html", games=config.GAMES, responses={}, selected_models=[]
    )


@app.route("/results", methods=["GET"])
def display_results():
    """Display results in a structured HTML table."""
    limit = request.args.get("limit", default=10, type=int)
    results = fetch_recent_results(limit)

    if not results:
        return "<h3>No results found.</h3>"

    return render_template("results.html", results=results)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
