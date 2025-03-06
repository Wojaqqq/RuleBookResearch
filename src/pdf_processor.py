from pathlib import Path
from pypdf import PdfReader
from config import Config
import re

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
