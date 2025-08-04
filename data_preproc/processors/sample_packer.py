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
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_tokens = config.get("min_tokens", 100000)
        self.max_tokens = config.get("max_tokens", 128000)
        self.text_field = config.get("text_field", "text")
        self.separator = config.get("separator", "\n\n")
        self.tokenizer_name = config.get("tokenizer", "gpt2")
        
        # Initialize tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        logger.info(f"Initialized SamplePackerProcessor with {self.min_tokens}-{self.max_tokens} token range")
        
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Not used - this processor needs access to the full dataset."""
        raise NotImplementedError("SamplePackerProcessor requires full dataset access. Use process_dataset instead.")
    
    def get_required_columns(self) -> List[str]:
        """Return required columns."""
        return [self.text_field]
        
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Pack samples together to reach target token lengths."""
        logger.info(f"Packing {len(dataset)} samples to reach {self.min_tokens}-{self.max_tokens} tokens")
        
        packed_samples = []
        current_texts = []
        current_token_count = 0
        samples_processed = 0
        
        # Pre-calculate separator tokens once
        separator_tokens = len(self.tokenizer.encode(self.separator)) - 1  # -1 for BOS token
        
        for i, sample in enumerate(dataset):
            text = sample.get(self.text_field, "")
            if not text:
                continue
                
            # Tokenize and get length
            text_tokens = len(self.tokenizer.encode(text))
            samples_processed += 1
            
            # Check if adding this sample would exceed max tokens
            additional_tokens = text_tokens
            if current_texts:  # Add separator tokens if not first sample
                additional_tokens += separator_tokens
            
            if current_token_count + additional_tokens > self.max_tokens:
                # Save current pack if it meets minimum threshold
                if current_token_count >= self.min_tokens:
                    packed_text = self.separator.join(current_texts)
                    packed_samples.append({self.text_field: packed_text})
                    if len(packed_samples) % 100 == 0:
                        logger.info(f"Created {len(packed_samples)} packs so far...")
                
                # Start new pack with current sample
                current_texts = [text]
                current_token_count = text_tokens
            else:
                # Add to current pack
                if current_texts:
                    current_token_count += separator_tokens
                current_texts.append(text)
                current_token_count += text_tokens
            
            # Log progress
            if samples_processed % 10000 == 0:
                logger.info(f"Processed {samples_processed} samples, created {len(packed_samples)} packs")
        
        # IMPORTANT: Handle remaining samples at the end
        # If we have accumulated samples, create final pack even if below min_tokens
        if current_texts:
            if current_token_count >= self.min_tokens:
                # Normal case: meets minimum threshold
                packed_text = self.separator.join(current_texts)
                packed_samples.append({self.text_field: packed_text})
                logger.info(f"Created final pack with {current_token_count} tokens")
            else:
                # Below minimum: try to add to last pack if possible
                if packed_samples and current_token_count > 0:
                    # Get the last pack and check if we can add to it
                    last_pack = packed_samples[-1][self.text_field]
                    last_pack_tokens = len(self.tokenizer.encode(last_pack))
                    
                    if last_pack_tokens + separator_tokens + current_token_count <= self.max_tokens:
                        # Add to last pack
                        packed_samples[-1][self.text_field] = last_pack + self.separator + self.separator.join(current_texts)
                        logger.info(f"Added {len(current_texts)} remaining samples to last pack")
                    else:
                        # Create a new pack even though it's below minimum
                        packed_text = self.separator.join(current_texts)
                        packed_samples.append({self.text_field: packed_text})
                        logger.info(f"Created final pack with {current_token_count} tokens (below minimum)")
                else:
                    # No previous packs, create one anyway
                    packed_text = self.separator.join(current_texts)
                    packed_samples.append({self.text_field: packed_text})
                    logger.info(f"Created single pack with {current_token_count} tokens")
        
        logger.info(f"Packing complete: {len(packed_samples)} packed samples from {samples_processed} original samples")
        logger.info(f"Packing ratio: {samples_processed / len(packed_samples):.1f}:1")
        
        # Create new dataset from packed samples
        return Dataset.from_list(packed_samples)


# Register the processor
register_processor("sample_packer", SamplePackerProcessor)