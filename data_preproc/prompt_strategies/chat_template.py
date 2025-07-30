"""Chat template prompt strategy for conversation-style datasets"""

from typing import Any, Dict, List, Optional

from data_preproc.utils.logging import get_logger

LOG = get_logger(__name__)


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    """
    Load chat template prompt strategy.

    Args:
        tokenizer: Tokenizer instance with chat template support
        cfg: Main configuration
        ds_cfg: Dataset configuration

    Returns:
        Function to process dataset examples
    """
    
    def tokenize_fn(examples):
        """Tokenize conversation examples using chat template"""
        # Check if tokenizer has chat template
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support chat templates")
        
        # Handle both single and batched examples
        if "messages" in examples:
            messages_list = examples["messages"]
        elif "conversations" in examples:
            messages_list = examples["conversations"]
        else:
            raise ValueError("Dataset must have 'messages' or 'conversations' field")
        
        if isinstance(messages_list, list) and messages_list and isinstance(messages_list[0], list):
            # Batched
            results = {"input_ids": [], "attention_mask": [], "labels": []}
            
            for messages in messages_list:
                # Apply chat template
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                
                # Tokenize
                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=cfg.sequence_len,
                    padding=False,
                    return_tensors=None,
                )
                
                # Add EOS token if needed
                if tokenized["input_ids"][-1] != tokenizer.eos_token_id:
                    tokenized["input_ids"].append(tokenizer.eos_token_id)
                    tokenized["attention_mask"].append(1)
                
                # Create labels
                labels = tokenized["input_ids"].copy()
                
                # Optionally mask system/user messages if train_on_inputs is False
                if not cfg.get("train_on_inputs", False):
                    # Simple heuristic: mask everything before the last assistant message
                    # This is a simplified version - more sophisticated masking can be added
                    labels = mask_user_messages(tokenizer, text, labels)
                
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                results["labels"].append(labels)
                
            return results
        else:
            # Single example
            messages = messages_list
            
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            # Tokenize
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=cfg.sequence_len,
                padding=False,
                return_tensors=None,
            )
            
            # Add EOS token if needed
            if tokenized["input_ids"][-1] != tokenizer.eos_token_id:
                tokenized["input_ids"].append(tokenizer.eos_token_id)
                tokenized["attention_mask"].append(1)
            
            # Create labels
            labels = tokenized["input_ids"].copy()
            
            # Optionally mask system/user messages
            if not cfg.get("train_on_inputs", False):
                labels = mask_user_messages(tokenizer, text, labels)
            
            tokenized["labels"] = labels
            return tokenized

    return tokenize_fn


def mask_user_messages(tokenizer, text: str, labels: List[int]) -> List[int]:
    """
    Simple function to mask user messages in labels.
    This is a basic implementation - more sophisticated versions can be added.
    """
    # For now, just return labels as-is
    # A proper implementation would parse the text to identify assistant vs user portions
    return labels