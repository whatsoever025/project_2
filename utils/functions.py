import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Tuple

def set_seed(seed: int = 42):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    This function sets the seed for Python's `random` module, NumPy, and PyTorch.
    It also configures PyTorch to ensure deterministic behavior in computations,
    which is particularly useful for debugging and reproducibility in machine
    learning experiments.

    Args:
        seed (int, optional): The seed value to use for random number generation.
                              Defaults to 42.

    Notes:
        - Setting `torch.backends.cudnn.deterministic` to `True` ensures that
          convolution operations are deterministic, but may reduce performance.
        - Setting `torch.backends.cudnn.benchmark` to `False` disables the
          auto-tuner that selects the best algorithm for the hardware, which
          also helps with reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility
    torch.backends.cudnn.benchmark = False  # Disables auto-tuning

TOKENIZER = None

def load_dataset_splits() -> Tuple[dict, dict, dict]:
    """
    Loads the CNN/DailyMail dataset splits for training, validation, and testing.

    This function uses the `datasets` library to load the CNN/DailyMail dataset
    (version 3.0.0) from the Hugging Face Hub. It returns the train, validation,
    and test splits as dictionaries.

    Returns:
        Tuple[dict, dict, dict]: A tuple containing the train, validation, and test
        splits of the dataset.

    Notes:
        - Requires the `datasets` library to be installed.
        - The dataset is expected to have 'article' and 'highlights' fields.
    """
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    return dataset["train"], dataset["validation"], dataset["test"]

def tokenize_t5(example, tokenizer, max_input_length: int = 512, max_target_length: int = 128):
    """
    Tokenizes input articles and target summaries for abstractive summarization using T5-style formatting.

    This function tokenizes the article and highlights (summary) from the dataset,
    ensuring that the input and target sequences are truncated to the specified
    maximum lengths. The tokenized inputs are prepared for model training, with
    labels set to the tokenized summaries.

    Args:
        example (dict): A dictionary containing:
            - "article" (str): The input article text.
            - "highlights" (str): The target summary text.
        max_input_length (int, optional): Maximum length for the input article.
                                         Defaults to 512.
        max_target_length (int, optional): Maximum length for the target summary.
                                          Defaults to 128.

    Returns:
        dict: A dictionary containing:
            - "input_ids" (list of int): Tokenized article input IDs.
            - "attention_mask" (list of int): Attention mask for the article.
            - "labels" (list of int): Tokenized summary IDs.
    """
    input_text = "summarize: " + example["article"]

    # Tokenize the article
    model_inputs = tokenizer(
        input_text,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Tokenize the summary (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["highlights"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    # Remove the tensor wrapping for compatibility with Dataset
    model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0).tolist()
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0).tolist()
    model_inputs["labels"] = labels["input_ids"].squeeze(0).tolist()

    return model_inputs

def tokenize_bart(example, tokenizer, max_input_length=512, max_target_length=128):
    """
    Tokenizes input/target using BART-style formatting.
    """
    model_inputs = tokenizer(
        example["article"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["highlights"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0).tolist()
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0).tolist()
    model_inputs["labels"] = labels["input_ids"].squeeze(0).tolist()

    return model_inputs

def prepare_dataset(
    tokenizer_name: str,
    model_type: str = "t5",
    max_input_length: int = 512,
    max_target_length: int = 128
) -> Tuple[dict, dict, dict, AutoTokenizer]:
    """
    Prepares the CNN/DailyMail dataset for abstractive summarization.

    This function loads the dataset, initializes the tokenizer, and tokenizes the
    train, validation, and test splits. It returns the processed datasets and the
    tokenizer.

    Args:
        tokenizer_name (str): The name or path of the tokenizer to be loaded.
        max_input_length (int, optional): Maximum length for the input article.
                                         Defaults to 512.
        max_target_length (int, optional): Maximum length for the target summary.
                                          Defaults to 128.

    Returns:
        Tuple[dict, dict, dict, AutoTokenizer]: A tuple containing the processed
        training dataset, validation dataset, testing dataset, and the loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load dataset splits
    train_data, val_data, test_data = load_dataset_splits()

    # train_size = int(0.01 * len(train_data))
    train_size = 500  
    val_size = int(0.02 * len(val_data))      
    test_size = int(0.02 * len(test_data)) 
    train_indices = random.sample(range(len(train_data)), train_size)
    val_indices = random.sample(range(len(val_data)), val_size)
    test_indices = random.sample(range(len(test_data)), test_size)
    train_data = train_data.select(train_indices)
    val_data = val_data.select(val_indices)
    test_data = test_data.select(test_indices)

    if model_type == "t5":
        tokenize_fn = lambda ex: tokenize_t5(ex, tokenizer, max_input_length, max_target_length)
    elif model_type == "bart":
        tokenize_fn = lambda ex: tokenize_bart(ex, tokenizer, max_input_length, max_target_length)

    # Tokenize datasets
    train_dataset = train_data.map(
        tokenize_fn,
        batched=False,
        remove_columns=train_data.column_names,
        num_proc=os.cpu_count() - 1
    )
    val_dataset = val_data.map(
        tokenize_fn,
        batched=False,
        remove_columns=val_data.column_names,
        num_proc=os.cpu_count() - 1
    )
    test_dataset = test_data.map(
        tokenize_fn,
        batched=False,
        remove_columns=test_data.column_names,
        num_proc=os.cpu_count() - 1
    )

    return train_dataset, val_dataset, test_dataset, tokenizer