from flask import Flask, request, jsonify
from rulebook_chatbot import RulebookChatbot

chatbot = RulebookChatbot()
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    """API endpoint to chat about a specififed board game."""
    data = request.json
    game_name = data.get("game")
    user_query = data.get("query")

    if not game_name or not user_query:
        return jsonify({"error": "Missing 'game' or 'query' in request."}), 400

    response = chatbot.ask_chatgpt(game_name, user_query)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
