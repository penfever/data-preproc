"""HuggingFace Datasets filter processor that preserves original structure."""

from typing import Dict, Any, Optional, List
import logging
from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class HFFilterProcessor(DatasetProcessor):
    """Apply HuggingFace Datasets .filter() method with tokenization for length checks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_tokens = config.get("max_tokens")
        self.min_tokens = config.get("min_tokens")
        self.filter_corrupted_images = config.get("filter_corrupted_images", False)
        self.max_image_size = config.get("max_image_size")
        self.min_image_size = config.get("min_image_size")
        self.tokenizer = config.get("tokenizer")  # Will be set by the caller
        
        # Text field extraction preferences
        self.text_fields = config.get("text_fields", ["problem", "solution", "question", "answer", "text", "content"])
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """This won't be used - we override the dataset processing."""
        return example
    
    def apply_to_dataset(self, dataset):
        """Apply filtering directly to the HF dataset using .filter() method."""
        
        # Track filter statistics for better error reporting
        filter_stats = {
            'token_count_too_low': 0,
            'token_count_too_high': 0,
            'no_text_content': 0,
            'corrupted_image': 0,
            'image_size_invalid': 0,
            'total_processed': 0,
            'passed': 0,
            'multiple_failures': 0
        }
        
        def filter_function(example):
            """Filter function that gets applied to each example."""
            filter_stats['total_processed'] += 1
            
            # Track all failing conditions for this example
            failed = False
            failure_count = 0
            
            # Check token length if tokenizer is available
            if self.tokenizer and (self.max_tokens or self.min_tokens):
                text_content = self._extract_text_content(example)
                if text_content:
                    # Tokenize to check length
                    tokens = self.tokenizer(text_content, add_special_tokens=False)
                    token_count = len(tokens["input_ids"])
                    
                    if self.max_tokens and token_count > self.max_tokens:
                        LOG.debug(f"Filtering: token count {token_count} > {self.max_tokens}")
                        filter_stats['token_count_too_high'] += 1
                        failed = True
                        failure_count += 1
                    
                    if self.min_tokens and token_count < self.min_tokens:
                        LOG.debug(f"Filtering: token count {token_count} < {self.min_tokens}")
                        filter_stats['token_count_too_low'] += 1
                        failed = True
                        failure_count += 1
                else:
                    # No text content found
                    LOG.debug("Filtering: no text content found")
                    filter_stats['no_text_content'] += 1
                    failed = True
                    failure_count += 1
            
            # Check image corruption
            if self.filter_corrupted_images and "image" in example:
                if not self._validate_image(example["image"]):
                    LOG.debug("Filtering: corrupted image")
                    filter_stats['corrupted_image'] += 1
                    failed = True
                    failure_count += 1
            
            # Check image size constraints
            if ("image" in example and example["image"] and 
                (self.max_image_size or self.min_image_size)):
                if not self._check_image_size(example["image"]):
                    filter_stats['image_size_invalid'] += 1
                    failed = True
                    failure_count += 1
            
            if failure_count > 1:
                filter_stats['multiple_failures'] += 1
            
            if not failed:
                filter_stats['passed'] += 1
                return True
            else:
                return False
        
        # Apply the filter using HF Datasets .filter() method
        initial_count = len(dataset)
        LOG.info(f"🔍 HF Filter: Processing {initial_count} examples")
        LOG.info(f"  Token limits: min={self.min_tokens}, max={self.max_tokens}")
        LOG.info(f"  Image checks: corrupted={self.filter_corrupted_images}, size_limits={bool(self.max_image_size or self.min_image_size)}")
        
        filtered_dataset = dataset.filter(filter_function)
        final_count = len(filtered_dataset)
        filtered_count = initial_count - final_count
        
        # Log detailed statistics
        LOG.info(f"📊 HF Filter Results:")
        LOG.info(f"  ✅ Passed: {filter_stats['passed']}/{initial_count} ({filter_stats['passed']/initial_count*100:.1f}%)")
        LOG.info(f"  ❌ No text content found: {filter_stats['no_text_content']}")
        LOG.info(f"  ❌ Token count too low (<{self.min_tokens}): {filter_stats['token_count_too_low']}")
        LOG.info(f"  ❌ Token count too high (>{self.max_tokens}): {filter_stats['token_count_too_high']}")
        LOG.info(f"  ❌ Corrupted images: {filter_stats['corrupted_image']}")
        LOG.info(f"  ❌ Invalid image size: {filter_stats['image_size_invalid']}")
        if filter_stats['multiple_failures'] > 0:
            LOG.info(f"  ⚠️  Examples with multiple failures: {filter_stats['multiple_failures']}")
        LOG.info(f"  📉 Total filtered: {filtered_count}/{initial_count} ({filtered_count/initial_count*100:.1f}%)")
        
        if final_count == 0 and initial_count > 0:
            # Log detailed filter reasons when all examples are filtered
            reasons = []
            if filter_stats['token_count_too_low'] > 0:
                reasons.append(f"token_count_too_low (<{self.min_tokens}): {filter_stats['token_count_too_low']}")
            if filter_stats['token_count_too_high'] > 0:
                reasons.append(f"token_count_too_high (>{self.max_tokens}): {filter_stats['token_count_too_high']}")
            if filter_stats['corrupted_image'] > 0:
                reasons.append(f"corrupted_image: {filter_stats['corrupted_image']}")
            if filter_stats['image_size_invalid'] > 0:
                reasons.append(f"image_size_invalid: {filter_stats['image_size_invalid']}")
            
            LOG.error(f"⚠️  HF filter removed all examples. Reasons: {'; '.join(reasons)}")
            LOG.error(f"Current settings: min_tokens={self.min_tokens}, max_tokens={self.max_tokens}")
            LOG.error("💡 Suggestion: Adjust min_tokens/max_tokens settings based on your dataset's characteristics.")
        
        return filtered_dataset
    
    def _extract_text_content(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract text content from example for tokenization."""
        texts = []
        
        for field in self.text_fields:
            if field in example and example[field]:
                texts.append(str(example[field]))
        
        return " ".join(texts) if texts else None
    
    def _validate_image(self, image) -> bool:
        """Validate image integrity."""
        if not HAS_PIL:
            return True
        
        try:
            if hasattr(image, "verify"):
                img_copy = image.copy()
                try:
                    img_copy.verify()
                    return True
                finally:
                    # Explicitly clean up the copy
                    if hasattr(img_copy, 'close'):
                        img_copy.close()
                    del img_copy
            return True
        except Exception as e:
            LOG.debug(f"Image validation failed: {e}")
            return False
    
    def _check_image_size(self, image) -> bool:
        """Check image size constraints."""
        if not hasattr(image, "size"):
            return True
        
        width, height = image.size
        
        if self.max_image_size:
            max_w, max_h = self.max_image_size
            if width > max_w or height > max_h:
                LOG.debug(f"Image too large: {width}x{height} > {max_w}x{max_h}")
                return False
        
        if self.min_image_size:
            min_w, min_h = self.min_image_size
            if width < min_w or height < min_h:
                LOG.debug(f"Image too small: {width}x{height} < {min_w}x{min_h}")
                return False
        
        return True
    
    def get_required_columns(self) -> List[str]:
        return []


# Register the processor
register_processor("hf_filter", HFFilterProcessor)