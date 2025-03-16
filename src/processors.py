from pathlib import Path
from pypdf import PdfReader
from config import Config
import re
import shutil
import faiss
import datetime
import openai
import faiss
import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()
config = Config.get_instance()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PDFProcessor:
    def __init__(self):
        self.pdf_folder = Path(config.PDF_FOLDER)
        self.gt_folder = config.GT_FOLDER
        self.extracted_folder = config.EXTRACTED_FOLDER

    def get_pdf_list(self):
        return [pdf for pdf in self.pdf_folder.glob("*.pdf")]

    def extract_text_from_pdfs(self):
        extracted_data = {}

        grouped_files = self.group_pdfs_by_base_game()

        for base_game, files in grouped_files.items():
            full_text = ""
            for pdf_path in files:
                part_text = self._get_text(pdf_path.stem)
                if part_text:
                    full_text += f"\n\n--- From {pdf_path.stem} ---\n\n" + part_text

            if full_text.strip():
                extracted_data[base_game] = full_text

        return extracted_data

    def group_pdfs_by_base_game(self):
        """
        Groups files like witcher_old_world_ciri.pdf under the base game 'witcher_old_world'.
        """
        pdf_list = self.get_pdf_list()
        grouped = {}

        base_game_pattern = re.compile(r"^(.*?)(?:_.*)?$")

        for pdf_path in pdf_list:
            base_game_match = base_game_pattern.match(pdf_path.stem)
            if base_game_match:
                base_game = base_game_match.group(1)

                if base_game not in grouped:
                    grouped[base_game] = []
                grouped[base_game].append(pdf_path)

        return grouped

    def _get_text(self, game_name):
        gt_text_file = self.gt_folder / f"{game_name}.txt"
        extracted_text_file = self.extracted_folder / f"{game_name}.txt"

        if gt_text_file.exists():
            return gt_text_file.read_text(encoding="utf-8").strip()

        if extracted_text_file.exists():
            return extracted_text_file.read_text(encoding="utf-8").strip()

        return self._extract_text_from_pdf(game_name, extracted_text_file)

    def _extract_text_from_pdf(self, game_name, extracted_text_file):
        pdf_path = self.pdf_folder / f"{game_name}.pdf"
        text = ""

        with pdf_path.open("rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        if text:
            extracted_text_file.write_text(text.strip(), encoding="utf-8")

        return text.strip()


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
        response = client.embeddings.create(
            input=[text], model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding)

    @staticmethod
    def chunk_text(text, chunk_size=800):
        """Splits text into smaller chunks."""
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


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
                dataset.append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": f"You are an expert on the board game {game_name} rules.",
                            },
                            {
                                "role": "user",
                                "content": "Explain the rules of this game.",
                            },
                            {"role": "assistant", "content": chunk},
                        ]
                    }
                )

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
                    print(
                        f"Continuing fine-tuning from existing fine-tuned model: {previous_model_id}"
                    )
                    starting_model = (
                        previous_model_id  # Use existing fine-tuned model as base
                    )

        # Submit fine-tuning job to OpenAI
        response = client.fine_tuning.jobs.create(
            training_file=file_response.id, model=starting_model
        )

        # Save new model ID to track for future runs
        with self.fine_tuned_model_path.open("w", encoding="utf-8") as f:
            json.dump({"model_id": response.id}, f)

        # Archive the dataset file
        archive_path = (
            self.archive_folder
            / f"fine_tune_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
        )
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
            print(
                "Error: fine_tune_dataset.jsonl not found. Run 'create-dataset' first."
            )
            return

        total_tokens = 0

        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                for message in example["messages"]:
                    total_tokens += (
                        len(message["content"].split()) + len(message["content"]) // 4
                    )

        cost_per_million_tokens = 3  # gpt-4o-mini as of February 2025
        total_cost = (total_tokens / 1_000_000) * cost_per_million_tokens

        print(f"Estimated fine-tuning cost for gpt-4o-mini:")
        print(f" - Total Tokens: {total_tokens}")
        print(f" - Estimated Cost: ${total_cost:.2f} USD")


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
            print("[DEBUG] No embeddings loaded.")
            return False, "", 0

        print(f"[DEBUG] Looking for game_name {game_name}")
        mapped_game_name = (
            config.EMBEDDING_MAPPING.get(game_name, game_name).strip().lower()
        )
        print(f"[DEBUG] Searching for game in mapped game name: {mapped_game_name}")

        query_embedding = self._get_embedding(query).astype("float32").reshape(1, -1)

        distances, indices = self.vector_store.search(query_embedding, 5)

        print(f"[DEBUG] Embedding search distances: {distances}")

        for idx, distance in zip(indices[0], distances[0]):
            entry = self.metadata[idx]
            if entry["game"].strip().lower() == mapped_game_name:
                print(f"[DEBUG] Found match at distance {distance}")
                return True, entry["text"]

        print(f"[DEBUG] No match found for {mapped_game_name}")
        return False, ""

    def _get_embedding(self, text):
        response = client.embeddings.create(
            input=[text], model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding)
