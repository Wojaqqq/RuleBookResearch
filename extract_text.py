from pathlib import Path
from utils import pdf_to_text_pymupdf

PDFS_PATH = Path("data/pdfs")
EXTRACTED_TXTS_PATH = Path("data/extracted_txts")


if __name__ == "__main__":

    pdf_name = Path("letters_from_whitechapel_rulebook")
    extracted_text = pdf_to_text_pymupdf(PDFS_PATH / f"{pdf_name}.pdf")

    if extracted_text is not None:
        with open(EXTRACTED_TXTS_PATH / f"{pdf_name.stem}.txt", "w", encoding='utf-8') as text_file:
            text_file.write(extracted_text)
