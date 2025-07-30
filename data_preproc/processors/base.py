"""Base dataset processors for common formats."""

from typing import Dict, Any, Optional, List
import logging
from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)


class PassThroughProcessor(DatasetProcessor):
    """Processor that passes data through unchanged."""
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pass through unchanged."""
        return example
    
    def get_required_columns(self) -> List[str]:
        """No specific requirements."""
        return []


class FilterProcessor(DatasetProcessor):
    """Processor that filters examples based on criteria."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_length = config.get("max_length")
        self.min_length = config.get("min_length") 
        self.required_fields = config.get("required_fields", [])
        self.filter_corrupted_images = config.get("filter_corrupted_images", False)
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter example based on criteria."""
        # Check required fields
        if not all(field in example for field in self.required_fields):
            LOG.debug(f"Filtering example: missing required fields {self.required_fields}")
            return None
        
        # Check text length if available
        text_content = self._extract_text_content(example)
        if text_content:
            text_len = len(text_content)
            if self.max_length and text_len > self.max_length:
                LOG.debug(f"Filtering example: text too long ({text_len} > {self.max_length})")
                return None
            if self.min_length and text_len < self.min_length:
                LOG.debug(f"Filtering example: text too short ({text_len} < {self.min_length})")
                return None
        
        # Check for corrupted images if requested
        if self.filter_corrupted_images and "image" in example:
            if not self._validate_image(example["image"]):
                LOG.debug("Filtering example: corrupted image")
                return None
        
        return example
    
    def _extract_text_content(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract text content from example for length checking."""
        # Try common text fields
        for field in ["text", "content", "problem", "question", "instruction"]:
            if field in example and isinstance(example[field], str):
                return example[field]
        
        # Try solution/answer fields
        for field in ["solution", "answer", "output", "response"]:
            if field in example and isinstance(example[field], str):
                return example[field]
                
        return None
    
    def _validate_image(self, image) -> bool:
        """Validate that image is not corrupted."""
        try:
            if hasattr(image, "verify"):
                image.verify()
                return True
            elif hasattr(image, "size"):
                # Basic size check
                return image.size[0] > 0 and image.size[1] > 0
        except Exception as e:
            LOG.debug(f"Image validation failed: {e}")
            return False
        
        return True
    
    def get_required_columns(self) -> List[str]:
        """Return required fields."""
        return self.required_fields


class ColumnMappingProcessor(DatasetProcessor):
    """Processor that maps column names."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.column_mapping = config.get("column_mapping", {})
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map column names according to configuration."""
        if not self.column_mapping:
            return example
        
        mapped_example = {}
        for old_name, new_name in self.column_mapping.items():
            if old_name in example:
                mapped_example[new_name] = example[old_name]
        
        # Keep unmapped columns
        for key, value in example.items():
            if key not in self.column_mapping:
                mapped_example[key] = value
        
        return mapped_example
    
    def get_required_columns(self) -> List[str]:
        """No specific requirements."""
        return []


# Register the base processors
register_processor("passthrough", PassThroughProcessor)
register_processor("filter", FilterProcessor)
register_processor("column_mapping", ColumnMappingProcessor)