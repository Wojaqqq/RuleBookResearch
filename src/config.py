from pathlib import Path

class Config:
    instance = None

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.DATA_DIR = self.BASE_DIR / "data"

        self.PDF_FOLDER = self.DATA_DIR / "pdfs"
        self.GT_FOLDER = self.DATA_DIR / "GT"
        self.EXTRACTED_FOLDER = self.DATA_DIR / "extracted"
        self.VECTOR_STORE_FILE = self.DATA_DIR / "vector_store.faiss"
        self.FINE_TUNED_MODEL_PATH = self.DATA_DIR / "fine_tuned_model.json"
        self.ARCHIVE_FOLDER = self.DATA_DIR / "fine_tune_archive"
        self.DB_PATH = self.DATA_DIR / "results.db"

        self.GAMES = [
            "Witcher Old World",
            "Game of Thrones",
            "Letters from Whitechapel",
            "Monopoly",
            "Saboteur",
            "Sid Meiers Civilization",
            "Civilization Through the Ages"
        ]

        self.EMBEDDING_MAPPING = {
            "Monopoly": "monopoly",
            "Letters from Whitechapel": "letters",
            "Game of Thrones": "game_of_thrones",
            "Witcher Old World": "witcher"
        }

    @classmethod
    def get_instance(cls):
        if not cls.instance:
            cls.instance = cls()
        return cls.instance
