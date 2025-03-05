import os
import json
import numpy as np
import openai
import faiss
import argparse
import shutil
import datetime
from dotenv import load_dotenv
from config import Config
from pdf_processor import PDFProcessor
from pathlib import Path

load_dotenv()
config = Config.get_instance()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbeddingProcessor:
    """Handles FAISS vector store creation, fully independent from fine-tuning."""
    
    EMBEDDING_METADATA_FILE = config.DATA_DIR / "embedding_metadata.json"
    
    def __init__(self):
        self.vector_store = None
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Loads embedding-specific metadata (completely separate from fine-tuning)."""
        if self.EMBEDDING_METADATA_FILE.exists():
            with self.EMBEDDING_METADATA_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def update_embeddings(self):
        """Generates embeddings directly from PDFs/GT, unrelated to fine-tuning."""
        pdf_processor = PDFProcessor()
        extracted_data = pdf_processor.extract_text_from_pdfs()

        if not extracted_data:
            print("No text found for embeddings! Please add PDFs/GT files.")
            return

        embeddings = []
        metadata = []

        for game_name, text in extracted_data.items():
            chunks = self.chunk_text(text)
            for chunk in chunks:
                embeddings.append(self._get_embedding(chunk))
                metadata.append({"game": game_name, "text": chunk})

        embeddings = np.array(embeddings).astype("float32")

        if self.vector_store is None:
            self.vector_store = faiss.IndexFlatL2(embeddings.shape[1])

        self.vector_store.add(embeddings)
        faiss.write_index(self.vector_store, str(config.VECTOR_STORE_FILE))

        with self.EMBEDDING_METADATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Embeddings created and stored in {config.VECTOR_STORE_FILE}")
        print(f"Embedding metadata saved to {self.EMBEDDING_METADATA_FILE}")

    def _get_embedding(self, text):
        """Generates embeddings using OpenAI."""
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        return np.array(response.data[0].embedding)

    @staticmethod
    def chunk_text(text, chunk_size=800):
        """Splits text into smaller chunks."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


class FineTuneProcessor:
    """Handles fine-tuning dataset creation and submission."""

    FINE_TUNE_DATASET_FILE = config.DATA_DIR / "fine_tune_dataset.jsonl"

    def __init__(self):
        self.archive_folder = config.ARCHIVE_FOLDER
        self.fine_tuned_model_path = config.FINE_TUNED_MODEL_PATH


    def create_fine_tuning_dataset(self):
        """Creates dataset for fine-tuning (no relation to embeddings)."""
        pdf_processor = PDFProcessor()
        extracted_data = pdf_processor.extract_text_from_pdfs()

        if not extracted_data:
            print("No text found for fine-tuning! Please add PDFs/GT files.")
            return

        dataset = []
        for game_name, text in extracted_data.items():
            chunks = EmbeddingProcessor.chunk_text(text)
            for chunk in chunks:
                dataset.append({
                    "messages": [
                        {"role": "system", "content": f"You are an expert on the board game {game_name} rules."},
                        {"role": "user", "content": "Explain the rules of this game."},
                        {"role": "assistant", "content": chunk}
                    ]
                })

        with self.FINE_TUNE_DATASET_FILE.open("w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example) + "\n")

        print(f"Fine-tuning dataset saved to {self.FINE_TUNE_DATASET_FILE}")


    def fine_tune(self):
        """Submits fine-tuning dataset to OpenAI, either starting fresh or continuing from the latest fine-tuned model."""

        if not self.FINE_TUNE_DATASET_FILE.exists():
            print("Fine-tuning dataset not found! Run 'create-dataset' first.")
            return

        # Upload fine-tune dataset file
        with self.FINE_TUNE_DATASET_FILE.open("rb") as f:
            file_response = client.files.create(file=f, purpose="fine-tune")

        # Load existing fine-tuned model ID (if any)
        starting_model = "gpt-4o-mini-2024-07-18"  # Default base model
        if self.fine_tuned_model_path.exists():
            with self.fine_tuned_model_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
                previous_model_id = metadata.get("model_id")

                if previous_model_id:
                    print(f"Continuing fine-tuning from existing fine-tuned model: {previous_model_id}")
                    starting_model = previous_model_id  # Use existing fine-tuned model as base

        # Submit fine-tuning job to OpenAI
        response = client.fine_tuning.jobs.create(
            training_file=file_response.id,
            model=starting_model
        )

        # Save new model ID to track for future runs
        with self.fine_tuned_model_path.open("w", encoding="utf-8") as f:
            json.dump({"model_id": response.id}, f)

        # Archive the dataset file
        archive_path = self.archive_folder / f"fine_tune_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
        self.archive_folder.mkdir(parents=True, exist_ok=True)
        shutil.move(self.FINE_TUNE_DATASET_FILE, archive_path)

        print(f"Fine-tuning job submitted. Dataset archived at {archive_path}")
        print(f"Fine-tune Job ID: {response.id}")


    def check_fine_tune_status(self):
        """Checks fine-tuning job status from OpenAI."""
        if not self.fine_tuned_model_path.exists():
            print("No fine-tuned model found. Run 'fine-tune' first.")
            return

        with self.fine_tuned_model_path.open("r", encoding="utf-8") as f:
            job_id = json.load(f).get("model_id")

        if not job_id:
            print("Invalid job_id in fine-tuned model metadata.")
            return

        response = client.fine_tuning.jobs.retrieve(job_id)

        print(f"Fine-tune job '{job_id}' is currently: {response.status.upper()}")
    
    def estimate_fine_tuning_cost(self):
        """Estimates the fine-tuning cost for the current dataset in fine_tune_dataset.jsonl."""

        dataset_path = self.FINE_TUNE_DATASET_FILE

        if not dataset_path.exists():
            print("Error: fine_tune_dataset.jsonl not found. Run 'create-dataset' first.")
            return

        total_tokens = 0

        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                for message in example["messages"]:
                    total_tokens += len(message["content"].split()) + len(message["content"]) // 4

        cost_per_million_tokens = 3  # gpt-4o-mini as of February 2025
        total_cost = (total_tokens / 1_000_000) * cost_per_million_tokens

        print(f"Estimated fine-tuning cost for gpt-4o-mini:")
        print(f" - Total Tokens: {total_tokens}")
        print(f" - Estimated Cost: ${total_cost:.2f} USD")


def print_help():
    """Prints help information about all modes."""
    print("""
Available modes:

make-embedding     - Create embeddings from PDFs/GT. Produces vector store & embedding metadata.

create-dataset     - Create dataset for fine-tuning only. Produces fine_tune_dataset.json.

fine-tune          - Submit fine-tuning dataset to OpenAI (works only with fine_tune_dataset.json).

check-status       - Check status of latest fine-tuning job.

estimate-cost      - Estimate the cost of fine-tuning based on current fine_tune_dataset.json.

help                - Show this help message.

Example usage:
python3 src/rulebook_processor.py create-dataset
python3 src/rulebook_processor.py make-embedding
python3 src/rulebook_processor.py fine-tune
python3 src/rulebook_processor.py check-status
python3 src/rulebook_processor.py estimate-cost
    """)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[
    "fine-tune", "make-embedding", "create-dataset", "check-status", "estimate-cost", "help"
    ], help="Choose operation mode")
    args = parser.parse_args()

    if args.mode == "make-embedding":
        processor = EmbeddingProcessor()
        processor.update_embeddings()

    elif args.mode == "create-dataset":
        processor = FineTuneProcessor()
        processor.create_fine_tuning_dataset()

    elif args.mode == "fine-tune":
        processor = FineTuneProcessor()
        processor.fine_tune()

    elif args.mode == "check-status":
        processor = FineTuneProcessor()
        processor.check_fine_tune_status()
    
    elif args.mode == "estimate-cost":
        processor = FineTuneProcessor()
        processor.estimate_fine_tuning_cost()

    elif args.mode == "help":
        print_help()
