#  RuleBook Research

A tool for creating embeddings and fine-tuning OpenAI models using board game rulebooks.

## Features

- Extract text from PDF rulebooks
- Create embeddings for semantic search of game rules
- Generate fine-tuning datasets from rulebook content
- Fine-tune OpenAI models with game rules
- Track multiple fine-tuned models and their capabilities
- Support for base games and expansions

## Folder Structure

```text
project_root/
├── src/                      # Python code
│   ├── processors/           # Processing modules for various tasks
│   │   ├── pdf_processor.py     # PDF text extraction
│   │   ├── embedding_processor.py # Embedding creation
│   │   ├── embedding_search.py   # Search through embeddings
│   │   ├── fine_tune_processor.py # Fine-tuning operations
│   │   ├── common.py            # Shared utilities
│   ├── embedai.py            # Main script for all operations
│   ├── model_info.py         # Tool for querying model capabilities
│   ├── config.py             # Configuration settings
├── data/                     # Data files
│   ├── pdfs/                 # Original rulebooks in PDF format
│   ├── extracted/            # Text extracted from PDFs (generated)
│   ├── GT/                   # Ground truth files (manual rule edits)
│   ├── vector_store.faiss    # Embedding index (generated)
│   ├── embedding_metadata.json # Metadata for embeddings (generated)
│   ├── fine_tuned_models_info.json # Tracks all fine-tuned models
│   ├── fine_tune_archive/    # Archives of past fine-tuning datasets
├── .env                      # Environment variables (create from template)
├── .env.template             # Sample environment file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
```

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.template` to `.env` and add your OpenAI API key
6. Create folders: `mkdir -p data/pdfs data/extracted data/GT data/fine_tune_archive`

## Usage

### Adding Rulebooks

Place PDF rulebooks in the `data/pdfs/` folder. For better organization:
- Use the format `game-name.pdf` for base games
- Use the format `game-name_expansion-name.pdf` for expansions

If you have hand-edited rules, place them in `data/GT/` with matching names.

### Creating Embeddings

```
python src/embedai.py make-embedding
```

This extracts text from PDFs and creates embeddings for semantic search.

### Creating and Submitting Fine-tuning Datasets

```
python src/embedai.py create-dataset
python src/embedai.py estimate-cost
python src/embedai.py fine-tune
```

When fine-tuning, you can select which base model to use and add a meaningful suffix.

### Checking Fine-tuning Status

```
python src/embedai.py check-status
```

This will check the status of all in-progress fine-tuning jobs. The system now properly handles both job IDs and model IDs, so you can track the entire lifecycle from submission to completion.

## File Naming Convention

Files follow a naming convention for base games and expansions:
- Base game: `game-name.pdf`
- Expansion: `game-name_expansion-name.pdf`

This helps the system understand relationships between rulebooks.

## Developer Notes

- The system now properly distinguishes between job IDs (used during fine-tuning) and model IDs (used for inference).
- Embedding metadata tracks included rulebooks and timestamps.
- When fine-tuning on top of an existing fine-tuned model, the system will update the existing entry rather than creating a new one.
- Environment variables are managed through the `.env` file.
- The project uses configuration values from `config.py` for consistency.

## Testing

To test your OpenAI API connection:

```
python src/embedai.py test-connection
```

To list available models:

```
python src/embedai.py list-models
```
