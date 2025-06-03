import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import os
from typing import Tuple, List, Union

# Add the parent directory to sys.path for potential custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define the cache directory within the project folder
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)  # Create cache directory if it doesn't exist

class BaseModelLoader:
    """Base class for model loaders."""
    def __init__(self, huggingface_repo: str, model_name: str):
        self.huggingface_repo = huggingface_repo
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """Load the model and tokenizer. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the load method.")

class BartModelLoader(BaseModelLoader):
    """Loader for BART-based models (uses BartForConditionalGeneration)."""
    def load(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        try:
            print(f"Loading BART model from {self.huggingface_repo}, subfolder: {self.model_name}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name,
                cache_dir=CACHE_DIR
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name,
                use_fast=True,
                cache_dir=CACHE_DIR
            )
            print(f"Successfully loaded BART model: {self.model_name}")
            return self.model, self.tokenizer
        except Exception as e:
            raise Exception(f"Error loading BART model '{self.model_name}': {str(e)}")

class T5ModelLoader(BaseModelLoader):
    """Loader for T5-based models (uses T5ForConditionalGeneration).
    Note: T5 requires 'summarize: ' prefix for summarization tasks in UI/inference.
    """
    def load(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        try:
            print(f"Loading T5 model from {self.huggingface_repo}, subfolder: {self.model_name}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name,
                cache_dir=CACHE_DIR
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name,
                use_fast=True,
                cache_dir=CACHE_DIR
            )
            print(f"Successfully loaded T5 model: {self.model_name}")
            return self.model, self.tokenizer
        except Exception as e:
            raise Exception(f"Error loading T5 model '{self.model_name}': {str(e)}")

class PegasusModelLoader(BaseModelLoader):
    """Loader for PEGASUS-based models (uses PegasusForConditionalGeneration)."""
    def load(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        try:
            print(f"Loading PEGASUS model from {self.huggingface_repo}, subfolder: {self.model_name}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name,
                cache_dir=CACHE_DIR
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name,
                use_fast=True,
                cache_dir=CACHE_DIR
            )
            print(f"Successfully loaded PEGASUS model: {self.model_name}")
            return self.model, self.tokenizer
        except Exception as e:
            raise Exception(f"Error loading PEGASUS model '{self.model_name}': {str(e)}")

class ModelLoaderFactory:
    """Factory class to manage model loaders and available models."""
    def __init__(self, huggingface_repo: str = "TheSyx/text-summarization"):
        self.huggingface_repo = huggingface_repo
        self.available_models = [
            "bart-experiment-1",  # Uses BartForConditionalGeneration
            "t5-experiment-1",   # Uses T5ForConditionalGeneration
            "pegasus-experiment-3"  # Uses PegasusForConditionalGeneration
        ]
        self.model_type_map = {
            "bart-experiment-1": "bart",
            "t5-experiment-1": "t5",
            "pegasus-experiment-3": "pegasus"
        }
        self.loader_map = {
            "bart": BartModelLoader,
            "t5": T5ModelLoader,
            "pegasus": PegasusModelLoader
        }

    def get_loader(self, model_name: str) -> BaseModelLoader:
        if model_name not in self.available_models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {self.available_models}"
            )
        
        model_type = self.model_type_map.get(model_name)
        if model_type not in self.loader_map:
            raise ValueError(f"Unsupported model type for '{model_name}': {model_type}")
        
        loader_class = self.loader_map[model_type]
        return loader_class(self.huggingface_repo, model_name)

    def get_available_models(self) -> List[str]:
        return self.available_models

if __name__ == "__main__":
    # Test loading all models and show working/failed models
    factory = ModelLoaderFactory()
    print("Testing loading of all models...")
    working_models = []
    failed_models = []

    for selected_model in factory.get_available_models():
        try:
            loader = factory.get_loader(selected_model)
            model, tokenizer = loader.load()
            working_models.append(selected_model)
        except Exception as e:
            failed_models.append(selected_model)
            print(f"Failed to load {selected_model}: {str(e)}")

    print("\nWorking models:")
    for model in working_models:
        print(f"- {model}")

    print("\nFailed models:")
    for model in failed_models:
        print(f"- {model}")

    print("\nFinished testing all models.")