import os
import logging
import PyPDF2
import fitz


def setup_logging(log_file_path):
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s',
        filemode='a'
    )

    logging.info("Logging setup complete")


def pdf_to_text_pypdf2(pdf_path):
    extracted_text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return extracted_text


def pdf_to_text_pymupdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        extracted_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_text += page.get_text("text")
        return extracted_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

