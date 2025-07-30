"""Completion prompt strategy for pretraining-style datasets"""

from typing import Any, Dict, Optional

from data_preproc.prompt_tokenizers import CompletionPromptTokenizingStrategy
from data_preproc.prompters import CompletionPrompter


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    """
    Load completion prompt strategy.

    Args:
        tokenizer: Tokenizer instance
        cfg: Main configuration
        ds_cfg: Dataset configuration

    Returns:
        Function to process dataset examples
    """
    prompter = CompletionPrompter()
    strategy = CompletionPromptTokenizingStrategy(
        prompter,
        tokenizer,
        train_on_inputs=True,  # For completion, we train on all tokens
        sequence_len=cfg.sequence_len,
    )

    def tokenize_fn(examples):
        """Tokenize examples for completion/pretraining"""
        # Handle both single and batched examples
        if "text" in examples:
            texts = examples["text"]
        elif "content" in examples:
            texts = examples["content"]
        else:
            # Try to find the first string column
            for key, value in examples.items():
                if isinstance(value, str) or (isinstance(value, list) and value and isinstance(value[0], str)):
                    texts = value
                    break
            else:
                raise ValueError("No text column found in dataset")

        if isinstance(texts, list):
            # Batched
            results = {"input_ids": [], "attention_mask": [], "labels": []}
            
            for text in texts:
                prompt_data = {"text": text}
                tokenized = strategy.tokenize_prompt(prompt_data)
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                results["labels"].append(tokenized["labels"])
                
            return results
        else:
            # Single example
            prompt_data = {"text": texts}
            return strategy.tokenize_prompt(prompt_data)

    return tokenize_fn