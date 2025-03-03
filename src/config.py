from pathlib import Path

class Config:
    instance = None

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.DATA_DIR = self.BASE_DIR / "../data"

        self.PDF_FOLDER = self.DATA_DIR / "pdfs"
        self.GT_FOLDER = self.DATA_DIR / "GT"
        self.EXTRACTED_FOLDER = self.DATA_DIR / "extracted"
        self.METADATA_FILE = self.DATA_DIR / "metadata.json"
        self.VECTOR_STORE_FILE = self.DATA_DIR / "vector_store.faiss"
        self.FINE_TUNED_MODEL_PATH = self.DATA_DIR / "fine_tuned_model.json"
        self.FINE_TUNED_METADATA_PATH = self.DATA_DIR / "fine_tuned_metadata.json"
        self.ARCHIVE_FOLDER = self.DATA_DIR / "fine_tune_archive"

    @classmethod
    def get_instance(cls):
        if not cls.instance:
            cls.instance = cls()
        return cls.instance
