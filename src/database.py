import os
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

try:
    connection_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=DATABASE_URL,
    )
    if not connection_pool:
        raise Exception("Error: Unable to create connection pool")
except Exception as e:
    raise Exception(f"Database connection pool initialization failed: {e}")


def get_db_connection():
    """Retrieve a connection from the pool."""
    try:
        return connection_pool.getconn()
    except Exception as e:
        raise Exception(f"Failed to get DB connection: {e}")


def close_db_connection(conn):
    """Return the connection back to the pool."""
    try:
        connection_pool.putconn(conn)
    except Exception as e:
        print(f"⚠ Warning: Could not return connection to pool: {e}")


def init_db():
    """Initialize the PostgreSQL database and create the 'results' table if not exists."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    question TEXT NOT NULL,
                    gpt4o_answer TEXT,
                    fine_tuned_answer TEXT,
                    embedding_answer TEXT,
                    gpt4o_tokens INTEGER DEFAULT 0,
                    fine_tuned_tokens INTEGER DEFAULT 0,
                    embedding_tokens INTEGER DEFAULT 0,
                    gpt4o_time FLOAT DEFAULT 0,
                    fine_tuned_time FLOAT DEFAULT 0,
                    embedding_time FLOAT DEFAULT 0,
                    chosen_model TEXT
                );
                """
            )
            conn.commit()
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        close_db_connection(conn)


def save_result(question, responses, token_counts, times, chosen_model):
    """Save the query and model responses to the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO results (
                    question, gpt4o_answer, fine_tuned_answer, embedding_answer, 
                    gpt4o_tokens, fine_tuned_tokens, embedding_tokens, 
                    gpt4o_time, fine_tuned_time, embedding_time, chosen_model
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    question,
                    responses.get("gpt4o", ""),
                    responses.get("fine_tuned", ""),
                    responses.get("embedding", ""),
                    token_counts.get("gpt4o", 0),
                    token_counts.get("fine_tuned", 0),
                    token_counts.get("embedding", 0),
                    times.get("gpt4o", 0.0),
                    times.get("fine_tuned", 0.0),
                    times.get("embedding", 0.0),
                    chosen_model,
                ),
            )
            conn.commit()
    except Exception as e:
        print(f"Error saving result: {e}")
    finally:
        close_db_connection(conn)


def fetch_recent_results(limit=5):
    """Fetch the most recent stored results from the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, timestamp, question, 
                       gpt4o_answer, fine_tuned_answer, embedding_answer, 
                       gpt4o_tokens, fine_tuned_tokens, embedding_tokens, 
                       gpt4o_time, fine_tuned_time, embedding_time, chosen_model
                FROM results
                ORDER BY timestamp DESC
                LIMIT %s;
                """,
                (limit,),
            )
            results = cur.fetchall()
        return results
    except Exception as e:
        print(f"Error fetching recent results: {e}")
        return []
    finally:
        close_db_connection(conn)
