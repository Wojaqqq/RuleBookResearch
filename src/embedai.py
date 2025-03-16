import argparse
import requests
import os
from dotenv import load_dotenv
from processors import EmbeddingProcessor, FineTuneProcessor

load_dotenv()


def print_help():
    """Prints help information and exit."""
    print(
        """
Available modes:

  make-embedding     - Create embeddings from PDFs/GT. Produces vector store & embedding metadata.

  create-dataset     - Create dataset for fine-tuning only. Produces fine_tune_dataset.json.

  fine-tune          - Submit fine-tuning dataset to OpenAI (works only with fine_tune_dataset.json).

  check-status       - Check status of latest fine-tuning job.

  estimate-cost      - Estimate the cost of fine-tuning based on current fine_tune_dataset.json.

  test-connection    - Check if OpenAI API is accessible with the provided API key.

  help               - Show this help message.

Example usage:
  python3 embedai.py create-dataset
  python3 embedai.py make-embedding
  python3 embedai.py fine-tune
  python3 embedai.py check-status
  python3 embedai.py estimate-cost
  python3 embedai.py test-connection
"""
    )
    exit(0)


def test_openai_connection():
    """Check OpenAI API connectivity and exit."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "API key not found! Please check your .env file or environment variables."
        )
        exit(1)

    response = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    if response.status_code == 200:
        print("OpenAI API connected!")
        exit(0)
    else:
        print(f"API error: {response.status_code} - {response.text}")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EmbedAI - A tool for embeddings, dataset creation, and fine-tuning openAI models."
    )
    parser.add_argument(
        "mode",
        choices=[
            "make-embedding",
            "create-dataset",
            "fine-tune",
            "check-status",
            "estimate-cost",
            "test-connection",
            "help",
        ],
        help="Choose an operation mode",
    )
    args = parser.parse_args()

    if args.mode == "help":
        print_help()

    if args.mode == "test-connection":
        test_openai_connection()

    embedding_processor = (
        EmbeddingProcessor() if args.mode == "make-embedding" else None
    )
    fine_tune_processor = (
        FineTuneProcessor()
        if args.mode in ["create-dataset", "fine-tune", "check-status", "estimate-cost"]
        else None
    )

    if args.mode == "make-embedding":
        embedding_processor.update_embeddings()

    elif args.mode == "create-dataset":
        fine_tune_processor.create_fine_tuning_dataset()

    elif args.mode == "fine-tune":
        fine_tune_processor.fine_tune()

    elif args.mode == "check-status":
        fine_tune_processor.check_fine_tune_status()

    elif args.mode == "estimate-cost":
        fine_tune_processor.estimate_fine_tuning_cost()
