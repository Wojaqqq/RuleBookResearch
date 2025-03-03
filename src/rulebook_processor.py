import os
import json
import numpy as np
import openai
import tiktoken
import faiss
import argparse
from dotenv import load_dotenv
import PyPDF2
import shutil
import datetime
from config import Config
load_dotenv()

config = Config.get_instance()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chunk_text(text, chunk_size=800):
    """Splits text into smaller chunks to fit model token limits."""
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk) for chunk in chunks]


class EmbeddingProcessor:
    def __init__(self, load_embeddings=True):
        self.vector_store = None
        self.metadata = self._load_metadata()
        if load_embeddings:
            self._update_vector_store_if_needed()
    
    
    def _load_metadata(self):
        if os.path.exists(config.METADATA_FILE):
            with open(config.METADATA_FILE, "r") as f:
                return json.load(f)
        return []

    def _update_vector_store_if_needed(self):
        existing_games = {entry["game"] for entry in self.metadata}
        new_files = [f for f in os.listdir(config.PDF_FOLDER) if f.endswith(".pdf") and f.replace(".pdf", "") not in existing_games]
        
        if not new_files:
            self._load_vector_store()
            return
        
        self._add_new_pdfs_to_vector_store(new_files)

    def _load_vector_store(self):
        if os.path.exists(config.VECTOR_STORE_FILE):
            self.vector_store = faiss.read_index(config.VECTOR_STORE_FILE)
        else:
            self.vector_store = faiss.IndexFlatL2(1536)
    
    def _add_new_pdfs_to_vector_store(self, new_files):
        embeddings = []
        texts = []

        for filename in new_files:
            game_name = filename.replace(".pdf", "")
            pdf_path = os.path.join(config.PDF_FOLDER, filename)
            text = self._load_or_extract_text(game_name, pdf_path)
            
            if text:
                chunks = chunk_text(text)
                for chunk in chunks:
                    embedding = self._get_embedding(chunk)
                    embeddings.append(embedding)
                    texts.append({"game": game_name, "text": chunk})
        
        if embeddings:
            embeddings = np.array(embeddings).astype("float32")
            
            if self.vector_store is None:
                self.vector_store = faiss.IndexFlatL2(embeddings.shape[1])
            
            self.vector_store.add(embeddings)
            faiss.write_index(self.vector_store, config.VECTOR_STORE_FILE)
            
            self.metadata.extend(texts)
            with open(config.METADATA_FILE, "w") as f:
                json.dump(self.metadata, f)
    
    def _load_or_extract_text(self, game_name, pdf_path):
        extracted_text_file = os.path.join(config.EXTRACTED_FOLDER, f"{game_name}.txt")
        
        if os.path.exists(extracted_text_file):
            with open(extracted_text_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        
        text = self._extract_text_from_pdf(pdf_path)
        
        if text:
            with open(extracted_text_file, "w", encoding="utf-8") as f:
                f.write(text)
        
        return text

    def _extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()
    
    def _get_embedding(self, text):
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        return np.array(response.data[0].embedding)


def create_fine_tuning_dataset():
    """Creates a fine-tuning dataset from new board game rulebooks only."""
    processor = EmbeddingProcessor(load_embeddings=False)
    
    # Load existing fine-tuned games
    fine_tuned_games = set()
    if os.path.exists(config.FINE_TUNED_METADATA_PATH):
        with open(config.FINE_TUNED_METADATA_PATH, "r") as f:
            fine_tuned_games = set(json.load(f).get("games", []))

    dataset = []
    for game in processor.metadata:
        game_name = game["game"]

        if game_name in fine_tuned_games:
            print(f"Skipping {game_name} - Already fine-tuned.")
            continue  # Skip games already fine-tuned

        chunks = [game["text"][i:i+800] for i in range(0, len(game["text"]), 800)]
        for chunk in chunks:
            dataset.append({
                "messages": [
                    {"role": "system", "content": f"You are an expert on the board game {game_name} rules."},
                    {"role": "user", "content": "Explain the rules of this game."},
                    {"role": "assistant", "content": chunk}
                ]
            })

    if not dataset:
        print("No new games to fine-tune!")
        return

    output_path = os.path.join("../data", "fine_tune_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"Fine-tuning dataset saved to {output_path}")


def fine_tune():
    dataset_path = os.path.join("../data", "fine_tune_dataset.json")
    
    if not os.path.exists(dataset_path):
        print("Fine-tuning dataset not found! Run 'create-dataset' mode first.")
        return

    # Upload dataset to OpenAI
    with open(dataset_path, "rb") as f:
        file_response = client.files.create(file=f, purpose="fine-tune")

    file_id = file_response.id  # Get the uploaded file ID

    # Check if we already have a fine-tuned model
    fine_tuned_model_id = None
    if os.path.exists(config.FINE_TUNED_MODEL_PATH):
        with open(config.FINE_TUNED_MODEL_PATH, "r") as f:
            fine_tuned_data = json.load(f)
            fine_tuned_model_id = fine_tuned_data.get("model_id")

    # Determine whether to create a new fine-tuned model or continue training an existing one
    if fine_tuned_model_id is None:
        print("No fine-tuned model found. Creating a new model...")
        response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model="gpt-4o-mini"
        )
    else:
        print(f"Existing fine-tuned model found: {fine_tuned_model_id}. Fine-tuning further...")
        response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=fine_tuned_model_id  # Continue training on the existing fine-tuned model
        )

    fine_tuned_model_id = response.id
    print(f"Fine-tuning job submitted: {fine_tuned_model_id}")

    with open(config.FINE_TUNED_MODEL_PATH, "w") as f:
        json.dump({"model_id": fine_tuned_model_id}, f)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = os.path.join(config.ARCHIVE_FOLDER, f"fine_tune_{timestamp}.json")

    os.makedirs(config.ARCHIVE_FOLDER, exist_ok=True)
    shutil.move(dataset_path, archive_path)

    print(f"Archived fine-tune dataset: {archive_path}")
    print(f"Fine-tuning completed! Model ID saved: {fine_tuned_model_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fine-tune", "make-embedding", "create-dataset"], help="Choose operation mode")
    args = parser.parse_args()

    if args.mode == "make-embedding":
        processor = EmbeddingProcessor()
    elif args.mode == "fine-tune":
        fine_tune()
    elif args.mode == "create-dataset":
        create_fine_tuning_dataset()
