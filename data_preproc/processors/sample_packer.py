"""Sample packing processor that combines multiple samples to reach target token lengths."""
import logging
from typing import Dict, List, Any, Optional
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from . import DatasetProcessor, register_processor

logger = logging.getLogger(__name__)


class SamplePackerProcessor(DatasetProcessor):
    """Processor that packs multiple samples together to reach target token lengths.
    
    This processor combines multiple text samples until reaching a target token range,
    useful for creating long-context training data from shorter samples.
    
    Note: This processor works differently from others - it processes the entire dataset
    at once rather than individual examples, since it needs to combine multiple samples.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_tokens = config.get("min_tokens", 100000)
        self.max_tokens = config.get("max_tokens", 128000)
        self.text_field = config.get("text_field", "text")
        self.separator = config.get("separator", "\n\n")
        self.tokenizer_name = config.get("tokenizer", "gpt2")
        self.max_samples = config.get("max_samples", 10000)
        
        # Initialize tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Not used - this processor needs access to the full dataset."""
        # This processor needs to be handled specially in the pipeline
        raise NotImplementedError("SamplePackerProcessor requires full dataset access. Use process_dataset instead.")
    
    def get_required_columns(self) -> List[str]:
        """Return required columns."""
        return [self.text_field]
        
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Pack samples together to reach target token lengths."""
        logger.info(f"Packing samples to reach {self.min_tokens}-{self.max_tokens} tokens")
        
        packed_samples = []
        current_texts = []
        current_token_count = 0
        samples_processed = 0
        
        for sample in dataset:
            if samples_processed >= self.max_samples:
                break
                
            text = sample.get(self.text_field, "")
            if not text:
                continue
                
            text_tokens = len(self.tokenizer.encode(text))
            
            # Check if adding this sample would exceed max tokens
            separator_tokens = len(self.tokenizer.encode(self.separator)) if current_texts else 0
            
            if current_token_count + text_tokens + separator_tokens > self.max_tokens:
                # Save current pack if it meets minimum threshold
                if current_token_count >= self.min_tokens:
                    packed_text = self.separator.join(current_texts)
                    packed_samples.append({self.text_field: packed_text})
                    logger.debug(f"Created pack with {current_token_count} tokens")
                
                # Start new pack
                current_texts = [text]
                current_token_count = text_tokens
            else:
                # Add to current pack
                if current_texts:
                    current_token_count += separator_tokens
                current_texts.append(text)
                current_token_count += text_tokens
            
            samples_processed += 1
            
            # Log progress
            if samples_processed % 100 == 0:
                logger.info(f"Processed {samples_processed} samples, created {len(packed_samples)} packs")
        
        # Handle remaining samples
        if current_texts and current_token_count >= self.min_tokens:
            packed_text = self.separator.join(current_texts)
            packed_samples.append({self.text_field: packed_text})
            logger.debug(f"Created final pack with {current_token_count} tokens")
        
        logger.info(f"Created {len(packed_samples)} packed samples from {samples_processed} original samples")
        
        # Create new dataset from packed samples
        return Dataset.from_list(packed_samples)


# Register the processor
register_processor("sample_packer", SamplePackerProcessor)