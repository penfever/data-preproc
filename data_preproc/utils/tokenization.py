"""Tokenization utilities for dataset preprocessing"""

from typing import Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from data_preproc.utils.logging import get_logger

LOG = get_logger(__name__)


def check_dataset_labels(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    num_examples: int = 5,
    text_only: bool = False,
) -> None:
    """
    Check and log dataset labels for debugging.

    Args:
        dataset: Dataset to check
        tokenizer: Tokenizer instance
        num_examples: Number of examples to display
        text_only: Whether to show only text (not token IDs)
    """
    LOG.info(f"Checking {num_examples} examples from dataset...")
    
    for idx in range(min(num_examples, len(dataset))):
        example = dataset[idx]
        
        if "input_ids" in example:
            input_ids = example["input_ids"]
            
            if text_only:
                # Decode and display text
                text = tokenizer.decode(input_ids, skip_special_tokens=False)
                LOG.info(f"\nExample {idx + 1}:\n{text}\n")
            else:
                # Display token IDs and decoded text
                LOG.info(f"\nExample {idx + 1}:")
                LOG.info(f"Input IDs: {input_ids[:50]}...")  # Show first 50 tokens
                
                # Decode text
                text = tokenizer.decode(input_ids, skip_special_tokens=False)
                LOG.info(f"Decoded text: {text[:200]}...")  # Show first 200 chars
                
                # Show special tokens
                if hasattr(tokenizer, "special_tokens_map"):
                    LOG.info(f"Special tokens: {tokenizer.special_tokens_map}")
        else:
            LOG.warning(f"Example {idx + 1} does not contain 'input_ids'")
            LOG.info(f"Available keys: {list(example.keys())}")
            
            # Try to display raw content
            for key, value in example.items():
                if isinstance(value, str):
                    LOG.info(f"{key}: {value[:200]}...")
                else:
                    LOG.info(f"{key}: {str(value)[:200]}...")