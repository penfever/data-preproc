"""Dataset format converters for vision language datasets."""

from typing import Dict, List, Any, Optional
import logging

LOG = logging.getLogger("data_preproc")


def convert_qa_to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Q&A format to messages format.
    
    Handles datasets with 'problem'/'solution' or 'question'/'answer' fields.
    """
    messages = []
    images = []
    
    # Determine field names
    question_field = None
    answer_field = None
    
    if "problem" in example and "solution" in example:
        question_field = "problem"
        answer_field = "solution"
    elif "question" in example and "answer" in example:
        question_field = "question"
        answer_field = "answer"
    elif "original_question" in example and "original_answer" in example:
        question_field = "original_question"
        answer_field = "original_answer"
    else:
        # If no recognized Q&A fields, return as-is
        return example
    
    # Build messages
    messages.append({
        "role": "user",
        "content": example[question_field]
    })
    messages.append({
        "role": "assistant", 
        "content": example[answer_field]
    })
    
    # Handle images
    if "image" in example and example["image"] is not None:
        images = [example["image"]]
    elif "images" in example and example["images"] is not None:
        images = example["images"] if isinstance(example["images"], list) else [example["images"]]
    
    # Build converted example
    converted = {
        "messages": messages,
        "images": images,
        "videos": [],
        "audios": []
    }
    
    # Preserve other fields as metadata
    metadata = {}
    for key, value in example.items():
        if key not in [question_field, answer_field, "image", "images"]:
            metadata[key] = value
    
    if metadata:
        converted["metadata"] = metadata
    
    return converted


def detect_and_convert_dataset_format(dataset) -> Any:
    """Detect dataset format and apply appropriate converter."""
    # Check first example
    if len(dataset) == 0:
        return dataset
    
    first_example = dataset[0]
    
    # Already in messages format
    if "messages" in first_example:
        LOG.info("Dataset already in messages format")
        return dataset
    
    # Q&A format
    if any(field in first_example for field in ["problem", "question", "original_question"]):
        LOG.info("Converting Q&A format dataset to messages format")
        # Remove original columns after conversion
        columns_to_remove = []
        for field in ["problem", "solution", "question", "answer", 
                      "original_question", "original_answer"]:
            if field in dataset.column_names:
                columns_to_remove.append(field)
        
        converted = dataset.map(
            convert_qa_to_messages, 
            desc="Converting to messages format",
            remove_columns=columns_to_remove
        )
        return converted
    
    # Unknown format
    LOG.warning(f"Unknown dataset format with fields: {list(first_example.keys())}")
    return dataset