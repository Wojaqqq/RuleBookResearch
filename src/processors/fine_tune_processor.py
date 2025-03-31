import json
import shutil
import datetime
from pathlib import Path
from .common import config, client, parse_game_name, chunk_text
from .pdf_processor import PDFProcessor


class FineTuneProcessor:
    """Handles fine-tuning dataset creation and submission."""

    FINE_TUNE_DATASET_FILE = config.DATA_DIR / "fine_tune_dataset.jsonl"
    FINE_TUNE_MODELS_INFO_FILE = config.DATA_DIR / "fine_tuned_models_info.json"

    def __init__(self):
        self.archive_folder = config.ARCHIVE_FOLDER
        self.models_info = self._load_models_info()
        self.current_dataset_info = {
            "included_rulebooks": [],
            "timestamp": datetime.datetime.now().isoformat(),
            "examples_count": 0,
        }

        PDFProcessor.ensure_folders_exist()

        if client is None:
            print(
                "WARNING: OpenAI client not initialized. Fine-tuning operations will not work."
            )

    def _load_models_info(self):
        """Load information about all fine-tuned models."""
        if self.FINE_TUNE_MODELS_INFO_FILE.exists():
            with self.FINE_TUNE_MODELS_INFO_FILE.open("r") as f:
                return json.load(f)
        return {"models": [], "last_used_model_id": None}

    def _save_models_info(self):
        """Save information about all fine-tuned models."""
        with self.FINE_TUNE_MODELS_INFO_FILE.open("w") as f:
            json.dump(self.models_info, f, indent=4)

    def create_fine_tuning_dataset(self):
        """Creates dataset for fine-tuning (no relation to embeddings)."""
        pdf_processor = PDFProcessor()
        extracted_data = pdf_processor.extract_text_from_pdfs()

        if not extracted_data:
            print("No text found for fine-tuning! Please add PDFs/GT files.")
            return

        dataset = []
        included_rulebooks = []

        for game_name, text in extracted_data.items():
            game_info = parse_game_name(game_name)
            included_rulebooks.append(game_name)

            system_prompt = (
                f"You are an expert on the board game {game_info['base_name']}"
            )
            if game_info["is_expansion"]:
                system_prompt += f" and its {game_info['expansion_name']} expansion"
            system_prompt += " rules."

            chunks = chunk_text(text)
            for chunk in chunks:
                dataset.append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": "Explain the rules of this game.",
                            },
                            {"role": "assistant", "content": chunk},
                        ]
                    }
                )

        with self.FINE_TUNE_DATASET_FILE.open("w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example) + "\n")

        self.current_dataset_info = {
            "included_rulebooks": included_rulebooks,
            "timestamp": datetime.datetime.now().isoformat(),
            "examples_count": len(dataset),
        }

        print(
            f"Fine-tuning dataset with {len(dataset)} examples saved to {self.FINE_TUNE_DATASET_FILE}"
        )
        print(f"Included rulebooks: {', '.join(included_rulebooks)}")

    def fine_tune(self):
        """Submits fine-tuning dataset to OpenAI, either starting fresh or continuing from an existing model."""

        if client is None:
            print("ERROR: OpenAI client not initialized. Cannot perform fine-tuning.")
            return

        if not self.FINE_TUNE_DATASET_FILE.exists():
            print("Fine-tuning dataset not found! Run 'create-dataset' first.")
            return

        if (
            self.current_dataset_info["examples_count"] == 0
            and self.FINE_TUNE_DATASET_FILE.exists()
        ):
            try:
                with self.FINE_TUNE_DATASET_FILE.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    rulebooks = set()
                    for line in lines:
                        example = json.loads(line)
                        system_content = example["messages"][0]["content"]
                        if "expert on the board game" in system_content:
                            game_name = system_content.split(
                                "expert on the board game "
                            )[1].split(" rules")[0]
                            rulebooks.add(game_name)

                    self.current_dataset_info = {
                        "included_rulebooks": list(rulebooks),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "examples_count": len(lines),
                    }
            except Exception as e:
                print(f"Warning: Could not extract dataset information: {e}")

        try:
            models = self._get_available_models()
            if not models:
                print(
                    "No valid models available for fine-tuning. Please check your API key."
                )
                return

            selected_model = self._select_fine_tune_model()
            if not selected_model:
                return

            print(
                "\nEnter a meaningful suffix for your model (optional, press Enter to skip):"
            )
            print("This will be appended to the model name for easier identification.")
            print("Example: 'monopoly-v1' or 'boardgames-2024'")
            model_suffix = input("> ").strip()

            with self.FINE_TUNE_DATASET_FILE.open("rb") as f:
                file_response = client.files.create(file=f, purpose="fine-tune")

            fine_tune_params = {
                "training_file": file_response.id,
                "model": selected_model,
            }

            if model_suffix:
                fine_tune_params["suffix"] = model_suffix

            response = client.fine_tuning.jobs.create(**fine_tune_params)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            archive_path = self.archive_folder / f"fine_tune_{timestamp}.jsonl"
            shutil.move(self.FINE_TUNE_DATASET_FILE, archive_path)

            model_info = {
                "model_id": response.id,
                "base_model": selected_model,
                "suffix": model_suffix if model_suffix else None,
                "timestamp": timestamp,
                "status": "submitted",
                "dataset_info": self.current_dataset_info,
                "archive_path": str(archive_path),
            }

            if selected_model in [m["model_id"] for m in self.models_info["models"]]:
                existing_model = next(
                    m
                    for m in self.models_info["models"]
                    if m["model_id"] == selected_model
                )
                existing_rulebooks = set(
                    existing_model.get("dataset_info", {}).get("included_rulebooks", [])
                )
                new_rulebooks = set(self.current_dataset_info["included_rulebooks"])
                combined_rulebooks = list(existing_rulebooks.union(new_rulebooks))

                model_info["dataset_info"] = {
                    "included_rulebooks": combined_rulebooks,
                    "timestamp": timestamp,
                    "examples_count": existing_model.get("dataset_info", {}).get(
                        "examples_count", 0
                    )
                    + self.current_dataset_info["examples_count"],
                }

            self.models_info["models"].append(model_info)
            self.models_info["last_used_model_id"] = response.id
            self._save_models_info()

            print(f"Fine-tuning job submitted. Dataset archived at {archive_path}")
            print(f"Fine-tune Job ID: {response.id}")
            print(f"Base model: {selected_model}")
            if model_suffix:
                print(f"Model suffix: {model_suffix}")

        except Exception as e:
            print(f"Error during fine-tuning: {e}")

    def _get_available_models(self):
        """Gets list of available models for fine-tuning."""
        try:
            response = client.models.list()
            available_models = [
                model.id
                for model in response.data
                if model.id.startswith("gpt-") and "instruct" not in model.id
            ]

            for model in self.models_info["models"]:
                if model.get("status") == "succeeded":
                    available_models.append(model.get("fine_tuned_model"))

            return available_models
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []

    def _select_fine_tune_model(self):
        """Selects a model for fine-tuning."""
        available_models = self._get_available_models()

        if not available_models:
            print("No models available for fine-tuning!")
            return None

        print("\nAvailable models for fine-tuning:")
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model}")

        last_used = self.models_info.get("last_used_model_id")
        if last_used:
            for model in self.models_info["models"]:
                if (
                    model["model_id"] == last_used
                    and model.get("status") == "succeeded"
                ):
                    last_used = model.get("fine_tuned_model")
                    break
            print(f"\nDefault: {last_used}")

        try:
            choice = input(
                "\nPlease select a model by entering its number, or press Enter to use the default:\n> "
            )
            if not choice and last_used:
                return last_used
            if choice:
                index = int(choice) - 1
                if 0 <= index < len(available_models):
                    return available_models[index]
            return available_models[0]
        except (ValueError, IndexError):
            print("Invalid selection. Using default model.")
            return available_models[0]

    def check_fine_tune_status(self):
        """Checks fine-tuning job status from OpenAI."""
        if client is None:
            print(
                "ERROR: OpenAI client not initialized. Cannot check fine-tune status."
            )
            return

        models_info = self._load_models_info()

        if not models_info["models"]:
            print("No fine-tuned models found. Run 'fine-tune' first.")
            return

        try:
            updated = False
            for model in models_info["models"]:
                if model["status"] not in ["succeeded", "failed", "cancelled"]:
                    try:
                        job_id = model.get("job_id", model["model_id"])
                        response = client.fine_tuning.jobs.retrieve(job_id)
                        model["status"] = response.status
                        if response.status == "succeeded":
                            model["fine_tuned_model"] = response.fine_tuned_model
                            models_info["last_used_model_id"] = model["model_id"]
                        updated = True

                        print(f"Model {model['model_id']} status: {response.status}")
                    except Exception as e:
                        print(f"Error checking model {model['model_id']}: {e}")

            if updated:
                self.models_info = models_info
                self._save_models_info()

            print("\nAll fine-tuned models:")
            for i, model in enumerate(models_info["models"]):
                print(f"{i+1}. ID: {model['model_id']}")
                print(f"   Status: {model['status']}")
                print(f"   Base model: {model.get('base_model', 'unknown')}")
                print(f"   Created: {model.get('timestamp', 'unknown')}")
                print(
                    f"   Rulebooks: {', '.join(model.get('dataset_info', {}).get('included_rulebooks', []))}"
                )
                print()

        except Exception as e:
            print(f"Error checking fine-tune status: {e}")

    def estimate_fine_tuning_cost(self):
        """Estimate the cost of fine-tuning based on the current dataset file."""
        if not self.FINE_TUNE_DATASET_FILE.exists():
            print("Fine-tuning dataset not found! Run 'create-dataset' first.")
            return

        with self.FINE_TUNE_DATASET_FILE.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            total_tokens = 0
            for line in lines:
                example = json.loads(line)
                for message in example["messages"]:
                    total_tokens += len(message["content"]) // 4

        # Calculate estimated costs for various models
        # These rates might need adjusting based on OpenAI's current pricing
        cost_rates = {
            "gpt-4o-mini": 0.00016,
            "gpt-4o": 0.00032,
            "gpt-4": 0.00060,
        }

        print(f"Dataset contains approximately {total_tokens} tokens")
        print("Estimated fine-tuning costs:")
        for model, rate in cost_rates.items():
            cost = total_tokens * rate
            print(f"  {model}: ${cost:.2f}")

        print("\nNote: These are rough estimates and actual costs may vary.")
