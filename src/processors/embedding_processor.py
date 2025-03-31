import json
import datetime
import numpy as np
import faiss
from .common import config, client, parse_game_name, chunk_text
from .pdf_processor import PDFProcessor


class EmbeddingProcessor:
    """Handles FAISS vector store creation, fully independent from fine-tuning."""

    EMBEDDING_METADATA_FILE = config.DATA_DIR / "embedding_metadata.json"

    def __init__(self):
        self.vector_store = None
        self.metadata = self._load_metadata()
        self.current_embedding_info = {
            "included_rulebooks": [],
            "timestamp": datetime.datetime.now().isoformat(),
            "chunks_count": 0,
        }
        if config.VECTOR_STORE_FILE.exists():
            self.vector_store = faiss.read_index(str(config.VECTOR_STORE_FILE))
        PDFProcessor.ensure_folders_exist()

    def _load_metadata(self):
        """Loads embedding-specific metadata (completely separate from fine-tuning)."""
        if self.EMBEDDING_METADATA_FILE.exists():
            with self.EMBEDDING_METADATA_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if "embedding_info" in data:
                    self.current_embedding_info = data["embedding_info"]
                return data
        return {"chunks": [], "embedding_info": self.current_embedding_info}

    def update_embeddings(self):
        """Generates embeddings directly from PDFs/GT, unrelated to fine-tuning."""
        pdf_processor = PDFProcessor()
        extracted_data = pdf_processor.extract_text_from_pdfs()

        if not extracted_data:
            print("No text found for embeddings! Please add PDFs/GT files.")
            return

        existing_chunks = self.metadata.get("chunks", [])
        existing_rulebooks = set(
            self.current_embedding_info.get("included_rulebooks", [])
        )
        existing_chunks_count = self.current_embedding_info.get("chunks_count", 0)

        new_embeddings = []
        new_metadata = []
        new_rulebooks = []

        for game_name, text in extracted_data.items():
            if game_name in existing_rulebooks:
                print(f"Skipping {game_name} - already in embeddings")
                continue

            game_info = parse_game_name(game_name)
            new_rulebooks.append(game_name)

            chunks = chunk_text(text)
            for chunk in chunks:
                new_embeddings.append(self._get_embedding(chunk))
                new_metadata.append(
                    {
                        "game": game_info["base_name"],
                        "full_name": game_name,
                        "is_expansion": game_info["is_expansion"],
                        "expansion_name": game_info["expansion_name"],
                        "text": chunk,
                    }
                )

        if not new_embeddings:
            print("No new rulebooks to add to embeddings!")
            return

        new_embeddings = np.array(new_embeddings).astype("float32")

        if self.vector_store is None:
            dimension = new_embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)

        self.vector_store.add(new_embeddings)
        faiss.write_index(self.vector_store, str(config.VECTOR_STORE_FILE))

        self.current_embedding_info = {
            "included_rulebooks": list(existing_rulebooks.union(new_rulebooks)),
            "timestamp": datetime.datetime.now().isoformat(),
            "chunks_count": existing_chunks_count + len(new_embeddings),
        }

        metadata_with_info = {
            "chunks": existing_chunks + new_metadata,
            "embedding_info": self.current_embedding_info,
        }

        with self.EMBEDDING_METADATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(metadata_with_info, f, indent=4)

        print(f"Embeddings updated and stored in {config.VECTOR_STORE_FILE}")
        print(f"Embedding metadata saved to {self.EMBEDDING_METADATA_FILE}")
        print(f"Added new rulebooks: {', '.join(new_rulebooks)}")
        print(
            f"Total rulebooks: {len(self.current_embedding_info['included_rulebooks'])}"
        )
        print(f"Total chunks: {self.current_embedding_info['chunks_count']}")

    def _get_embedding(self, text):
        """Generates embeddings using OpenAI."""
        response = client.embeddings.create(
            input=[text], model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding)
