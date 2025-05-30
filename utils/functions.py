import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from typing import Tuple
import pandas as pd

def set_seed(seed: int = 42):
    """
    Sets the random seed for various libraries to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TOKENIZER = None

def load_dataset_splits() -> Tuple[dict, dict, dict]:
    """
    Loads the CNN/DailyMail dataset splits for training, validation, and testing.
    """
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    return dataset["train"], dataset["validation"], dataset["test"]

def tokenize_t5(example, tokenizer, max_input_length: int = 512, max_target_length: int = 128):
    """
    Tokenizes input articles and target summaries for abstractive summarization using T5-style formatting.
    """
    input_text = "summarize: " + example["article"]
    model_inputs = tokenizer(
        input_text,
        max_length=max_input_length,
        truncation=True,  # Remove padding="max_length" for dynamic padding
        return_tensors="pt"
    )
    labels = tokenizer(
        text_target=example["highlights"],
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt"
    )
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
        truncation=True,
        return_tensors="pt"
    )
    labels = tokenizer(
        text_target=example["highlights"],
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt"
    )
    model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0).tolist()
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0).tolist()
    model_inputs["labels"] = labels["input_ids"].squeeze(0).tolist()
    return model_inputs

def tokenize_pegasus(example, tokenizer, max_input_length, max_target_length):
    model_inputs = tokenizer(
        example["article"],
        max_length=max_input_length,
        truncation=True,
        padding=False
    )
    labels = tokenizer(
        example["highlights"],
        max_length=max_target_length,
        truncation=True,
        padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return inputs

def prepare_dataset(
    tokenizer_name: str,
    model_type: str = "t5",
    max_input_length: int = 512,
    max_target_length: int = 128,
    save_dir: str = "./cnn_dailymail_subset_csv"
) -> Tuple[dict, dict, dict, AutoTokenizer]:
    """
    Prepares the CNN/DailyMail dataset for abstractive summarization by taking 20% of the initial data
    and resplitting it into train, validation, and test sets with an 8:1:1 ratio. If the subset exists
    as CSV files in save_dir, it loads the splits; otherwise, it creates them and saves as CSV files
    for manual upload to Hugging Face Hub.

    Args:
        tokenizer_name (str): The name or path of the tokenizer to be loaded.
        model_type (str, optional): Model type ("t5" or "bart"). Defaults to "t5".
        max_input_length (int, optional): Maximum length for the input article. Defaults to 512.
        max_target_length (int, optional): Maximum length for the target summary. Defaults to 128.
        save_dir (str, optional): Directory to save/load CSV files. Defaults to "./cnn_dailymail_subset_csv".

    Returns:
        Tuple[dict, dict, dict, AutoTokenizer]: A tuple containing the processed training dataset,
        validation dataset, testing dataset, and the loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define paths for CSV files
    train_csv = os.path.join(save_dir, "train.csv")
    val_csv = os.path.join(save_dir, "validation.csv")
    test_csv = os.path.join(save_dir, "test.csv")

    # Check if CSV files exist
    if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
        print(f"Loading dataset from CSV files in {save_dir}")
        train_data = Dataset.from_pandas(pd.read_csv(train_csv))
        val_data = Dataset.from_pandas(pd.read_csv(val_csv))
        test_data = Dataset.from_pandas(pd.read_csv(test_csv))
        print(f"Loaded dataset: Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")
    else:
        print(f"Creating and saving new dataset subset as CSV files in {save_dir}")

        # Load dataset splits
        train_data, val_data, test_data = load_dataset_splits()

        # Combine all splits into a single dataset
        full_dataset = concatenate_datasets([train_data, val_data, test_data])

        # Take 20% of the combined dataset
        total_size = len(full_dataset)
        subset_size = int(0.2 * total_size)
        subset_indices = random.sample(range(total_size), subset_size)
        subset_dataset = full_dataset.select(subset_indices)

        # Resplit into 8:1:1 (train:validation:test)
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
        train_size = int(train_ratio * subset_size)
        val_size = int(val_ratio * subset_size)
        test_size = subset_size - train_size - val_size

        # Shuffle indices and split
        shuffled_indices = random.sample(range(subset_size), subset_size)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]

        # Create new splits
        train_data = subset_dataset.select(train_indices)
        val_data = subset_dataset.select(val_indices)
        test_data = subset_dataset.select(test_indices)

        # Save splits as CSV files
        os.makedirs(save_dir, exist_ok=True)
        train_data.to_csv(train_csv)
        val_data.to_csv(val_csv)
        test_data.to_csv(test_csv)
        print(f"Dataset subset saved as CSV files to {save_dir}: Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

    # Select tokenization function based on model type
    if model_type == "t5":
        tokenize_fn = lambda ex: tokenize_t5(ex, tokenizer, max_input_length, max_target_length)
    elif model_type == "bart":
        tokenize_fn = lambda ex: tokenize_bart(ex, tokenizer, max_input_length, max_target_length)
    elif model_type == "pegasus":
        tokenize_fn = lambda ex: tokenize_pegasus(ex, tokenizer, max_input_length, max_target_length)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 't5' or 'bart'.")

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

if __name__ == "__main__":
    # Example usage
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(
        tokenizer_name="t5-base",
        model_type="t5",
        max_input_length=512,
        max_target_length=128,
        save_dir="./cnn_dailymail_subset_csv"
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")