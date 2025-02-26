import os
import json
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import faiss
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "../data/pdfs")
GT_FOLDER = os.path.join(BASE_DIR, "../data/GT")
EXTRACTED_FOLDER = os.path.join(BASE_DIR, "../data/extracted")
METADATA_FILE = os.path.join(BASE_DIR, "../data/metadata.json")
VECTOR_STORE_FILE = os.path.join(BASE_DIR, "../data/vector_store.faiss")


class RulebookChatbot:
    def __init__(self):
        """Initialize the chatbot and load or update the FAISS index."""
        self.vector_store = None
        self.game_rules = {}
        self.metadata = self._load_metadata()
        self._update_vector_store_if_needed()

    def _load_metadata(self):
        """Load metadata if it exists, otherwise return an empty list."""
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        return []

    def _update_vector_store_if_needed(self):
        """Check for new PDFs and update the FAISS index accordingly."""
        existing_games = {entry["game"] for entry in self.metadata}
        new_files = [
            f for f in os.listdir(PDF_FOLDER)
            if f.endswith(".pdf") and f.replace(".pdf", "") not in existing_games
        ]

        if not new_files:
            print("No new PDFs found. Loading existing FAISS index...")
            self._load_vector_store()
            return

        print(f"New PDFs detected: {new_files}. Updating FAISS index...")
        self._add_new_pdfs_to_vector_store(new_files)

    def _load_vector_store(self):
        """Load FAISS index if it exists."""
        if os.path.exists(VECTOR_STORE_FILE):
            self.vector_store = faiss.read_index(VECTOR_STORE_FILE)
        else:
            self.vector_store = faiss.IndexFlatL2(1536)  # OpenAI's embedding size

    def _add_new_pdfs_to_vector_store(self, new_files):
        """Process new PDFs and add their embeddings to FAISS."""
        embeddings = []
        texts = []

        for filename in new_files:
            game_name = filename.replace(".pdf", "")
            pdf_path = os.path.join(PDF_FOLDER, filename)
            text = self._load_or_extract_text(game_name, pdf_path)

            if text:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
                texts.append({"game": game_name, "text": text})

                self.game_rules[game_name] = text

        if embeddings:
            embeddings = np.array(embeddings).astype("float32")

            if self.vector_store is None:
                self.vector_store = faiss.IndexFlatL2(embeddings.shape[1])  # Create a new FAISS index

            self.vector_store.add(embeddings)  # Add new embeddings
            faiss.write_index(self.vector_store, VECTOR_STORE_FILE)  # Save updated FAISS index

            # Update metadata
            self.metadata.extend(texts)
            with open(METADATA_FILE, "w") as f:
                json.dump(self.metadata, f)

        print("FAISS index updated with new PDFs.")

    def _load_or_extract_text(self, game_name, pdf_path):
        """Load text from GT folder if available, otherwise extract from PDF."""
        gt_text_file = os.path.join(GT_FOLDER, f"{game_name}.txt")
        extracted_text_file = os.path.join(EXTRACTED_FOLDER, f"{game_name}.txt")

        if os.path.exists(gt_text_file):
            with open(gt_text_file, "r", encoding="utf-8") as f:
                print(f"Loading GT text for {game_name}...")
                return f.read().strip()

        if os.path.exists(extracted_text_file):
            with open(extracted_text_file, "r", encoding="utf-8") as f:
                print(f"Loading extracted text for {game_name}...")
                return f.read().strip()

        print(f"Extracting text from PDF: {game_name}")
        text = self._extract_text_from_pdf(pdf_path)

        if text:
            with open(extracted_text_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved extracted text to {extracted_text_file}")

        return text

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF."""
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()

    def _get_embedding(self, text):
        """Generate an embedding vector using OpenAI's latest API.
        
        Note: The new API requires the input to be a list of strings.
        """
        response = client.embeddings.create(input=[text],
        model="text-embedding-ada-002")
        return np.array(response.data[0].embedding)

    def search_relevant_text(self, query, game_name):
        """Find the most relevant rule snippet for a query.
        
        Returns:
            (bool, str): A tuple where the first element indicates if a relevant rule was found,
                         and the second element is the text snippet or an error message.
        """
        if not self.vector_store:
            return False, "No rulebooks found."

        embedding = self._get_embedding(query).astype("float32")
        _, idx = self.vector_store.search(np.array([embedding]), k=1)

        for i in idx[0]:
            if i < len(self.metadata) and self.metadata[i]["game"] == game_name:
                return True, self.metadata[i]["text"][:1000]

        return False, "No relevant rules found for this game."

    def ask_chatgpt(self, game_name, user_query):
        """Retrieve relevant rules and query ChatGPT while tracking token usage."""
        if game_name not in self.game_rules:
            return {"error": "Game not found in database."}

        found, relevant_text = self.search_relevant_text(user_query, game_name)

        if found:
            chat_response = client.chat.completions.create(model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are an expert on board game rules for {game_name}."},
                {"role": "user", "content": f"Here are the relevant rules: {relevant_text}\n\nQuestion: {user_query}"}
            ])

            response_text = chat_response.choices[0].message.content
            total_tokens = chat_response.usage.total_tokens

            return {
                "response": response_text,
                "tokens_used": total_tokens
            }
        else:
            return {"error": relevant_text}
