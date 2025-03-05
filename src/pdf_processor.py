from pathlib import Path
from pypdf import PdfReader
from config import Config

config = Config.get_instance()

class PDFProcessor:
    def __init__(self):
        self.pdf_folder = config.PDF_FOLDER
        self.gt_folder = config.GT_FOLDER
        self.extracted_folder = config.EXTRACTED_FOLDER

    def get_pdf_list(self):
        return [pdf for pdf in self.pdf_folder.glob("*.pdf")]

    def extract_text_from_pdfs(self):
        extracted_data = {}
        for pdf_path in self.get_pdf_list():
            game_name = pdf_path.stem
            text = self._get_text(game_name)
            if text:
                extracted_data[game_name] = text
        return extracted_data

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