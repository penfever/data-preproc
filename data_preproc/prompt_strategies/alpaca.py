"""Alpaca prompt strategy"""

from typing import Any, Dict, Optional

from data_preproc.prompt_tokenizers import AlpacaPromptTokenizingStrategy
from data_preproc.prompters import AlpacaPrompter


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    """
    Load Alpaca prompt strategy.

    Args:
        tokenizer: Tokenizer instance
        cfg: Main configuration
        ds_cfg: Dataset configuration

    Returns:
        Function to process dataset examples
    """
    prompter = AlpacaPrompter()
    strategy = AlpacaPromptTokenizingStrategy(
        prompter,
        tokenizer,
        train_on_inputs=cfg.train_on_inputs if hasattr(cfg, "train_on_inputs") else False,
        sequence_len=cfg.sequence_len,
    )

    def tokenize_fn(examples):
        """Tokenize examples using Alpaca strategy"""
        # Handle both single and batched examples
        if isinstance(examples["instruction"], list):
            # Batched
            results = {"input_ids": [], "attention_mask": [], "labels": []}
            
            for i in range(len(examples["instruction"])):
                prompt_data = {
                    "instruction": examples["instruction"][i],
                    "input": examples.get("input", [""] * len(examples["instruction"]))[i],
                    "output": examples.get("output", [""] * len(examples["instruction"]))[i],
                }
                
                tokenized = strategy.tokenize_prompt(prompt_data)
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                results["labels"].append(tokenized["labels"])
                
            return results
        else:
            # Single example
            prompt_data = {
                "instruction": examples["instruction"],
                "input": examples.get("input", ""),
                "output": examples.get("output", ""),
            }
            return strategy.tokenize_prompt(prompt_data)

    return tokenize_fn