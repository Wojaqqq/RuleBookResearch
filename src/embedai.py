import argparse
import requests
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from processors.embedding_processor import EmbeddingProcessor
from processors.fine_tune_processor import FineTuneProcessor
from processors.common import config

load_dotenv()


def print_help():
    """Prints help information and exit."""
    print(
        """
Available modes:

  make-embedding     - Create embeddings from PDFs/GT. Produces vector store & embedding metadata.

  create-dataset     - Create dataset for fine-tuning only. Produces fine_tune_dataset.json.

  fine-tune          - Submit fine-tuning dataset to OpenAI (works only with fine_tune_dataset.json).

  check-status       - Check status of all fine-tuning jobs.

  estimate-cost      - Estimate the cost of fine-tuning based on current fine_tune_dataset.json.

  test-connection    - Check if OpenAI API is accessible with the provided API key.

  list-models        - List all available models and previously fine-tuned models.

  help               - Show this help message.

Example usage:
  python3 embedai.py create-dataset
  python3 embedai.py make-embedding
  python3 embedai.py fine-tune
  python3 embedai.py check-status
  python3 embedai.py estimate-cost
  python3 embedai.py test-connection
  python3 embedai.py list-models
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

    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

        if response.status_code == 200:
            print("OpenAI API connected!")
            exit(0)
        else:
            print(f"API error: {response.status_code} - {response.text}")
            exit(1)
    except Exception as e:
        print(f"Connection error: {e}")
        exit(1)


def check_requirements():
    """Check if required folders exist."""
    if not config.PDF_DIR.exists():
        print(f"Error: PDF folder {config.PDF_DIR} does not exist!")
        print("Please create the folder and add PDF files before running the program.")
        exit(1)


def list_models():
    """List all available models and previously fine-tuned models."""
    fine_tune_processor = FineTuneProcessor()
    models = fine_tune_processor._get_available_models()

    print("Available base models:")
    base_models = [m for m in models if not m.startswith("ft:")]
    for i, model in enumerate(base_models):
        print(f"{i+1}. {model}")

    print("\nPreviously fine-tuned models:")
    ft_models = fine_tune_processor.models_info["models"]
    if not ft_models:
        print("  No fine-tuned models found.")
    else:
        for i, model in enumerate(ft_models):
            print(f"{i+1}. ID: {model['model_id']}")
            print(f"   Status: {model['status']}")
            print(f"   Base model: {model.get('base_model', 'unknown')}")
            print(f"   Created: {model.get('timestamp', 'unknown')}")
            print(
                f"   Rulebooks: {', '.join(model.get('dataset_info', {}).get('included_rulebooks', []))}"
            )
            print()

    exit(0)


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
            "list-models",
            "help",
        ],
        help="Choose an operation mode",
    )

    args = parser.parse_args()

    if args.mode == "help":
        print_help()

    if args.mode == "test-connection":
        test_openai_connection()

    if args.mode == "list-models":
        list_models()

    # Check folder requirements before proceeding
    if args.mode not in ["help", "test-connection"]:
        check_requirements()

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
