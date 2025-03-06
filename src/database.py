import sqlite3
import os
from config import Config
from datetime import datetime, timezone

config = Config.get_instance()

def init_db():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            gpt4o_answer TEXT,
            fine_tuned_answer TEXT,
            embedding_answer TEXT,
            gpt4o_tokens INTEGER,
            fine_tuned_tokens INTEGER,
            embedding_tokens INTEGER,
            gpt4o_time REAL,
            fine_tuned_time REAL,
            embedding_time REAL,
            chosen_model TEXT
        )
        """)

def save_result(question, responses, tokens, times, chosen_model):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.execute("""
            INSERT INTO results (timestamp, question, gpt4o_answer, fine_tuned_answer, embedding_answer,
                                gpt4o_tokens, fine_tuned_tokens, embedding_tokens,
                                gpt4o_time, fine_tuned_time, embedding_time, chosen_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, question,
                responses.get("gpt4o", ""), responses.get("fine_tuned", ""), responses.get("embedding", ""),
                tokens.get("gpt4o", 0), tokens.get("fine_tuned", 0), tokens.get("embedding", 0),
                times.get("gpt4o", 0), times.get("fine_tuned", 0), times.get("embedding", 0),
                chosen_model
            ))
    except Exception as e:
        print(f"[ERROR] Failed to save result to database: {e}")
