"""Regex transformation processor for text field modifications."""

import logging
import re
from typing import Dict, Any, List, Optional, Union

from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)


class RegexTransformProcessor(DatasetProcessor):
    """Processor that applies regex transformations to text fields.
    
    Supports:
    - Multiple regex patterns with replacement strings
    - Field-specific transformations
    - Named capture groups
    - Flags (case-insensitive, multiline, etc.)
    - Multiple transformations per field
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.transformations = config.get("transformations", [])
        self.default_flags = self._parse_flags(config.get("default_flags", ["MULTILINE"]))
        
        # Validate and compile regex patterns
        self.compiled_transformations = []
        for i, transform in enumerate(self.transformations):
            try:
                compiled_transform = self._compile_transformation(transform)
                self.compiled_transformations.append(compiled_transform)
            except Exception as e:
                LOG.error(f"Error compiling transformation {i}: {e}")
                raise ValueError(f"Invalid regex transformation {i}: {e}")
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply regex transformations to the example."""
        result = example.copy()
        
        for transform in self.compiled_transformations:
            field = transform["field"]
            
            if field not in result:
                LOG.debug(f"Field '{field}' not found in example, skipping transformation")
                continue
                
            field_value = result[field]
            if not isinstance(field_value, str):
                LOG.debug(f"Field '{field}' is not a string, skipping transformation")
                continue
            
            # Apply the transformation
            try:
                new_value = self._apply_transformation(field_value, transform)
                result[field] = new_value
                LOG.debug(f"Applied regex transformation to field '{field}'")
            except Exception as e:
                LOG.warning(f"Error applying transformation to field '{field}': {e}")
                # Continue with original value
        
        return result
    
    def _compile_transformation(self, transform: Dict[str, Any]) -> Dict[str, Any]:
        """Compile a transformation configuration."""
        field = transform.get("field")
        if not field:
            raise ValueError("Transformation must specify a 'field'")
        
        pattern = transform.get("pattern")
        if not pattern:
            raise ValueError("Transformation must specify a 'pattern'")
        
        replacement = transform.get("replacement", "")
        flags = self._parse_flags(transform.get("flags", []))
        
        # Combine default flags with transformation-specific flags
        combined_flags = self.default_flags | flags
        
        # Compile the regex pattern
        compiled_pattern = re.compile(pattern, combined_flags)
        
        return {
            "field": field,
            "pattern": compiled_pattern,
            "replacement": replacement,
            "original_pattern": pattern,
            "count": transform.get("count", 0)  # 0 means replace all
        }
    
    def _apply_transformation(self, text: str, transform: Dict[str, Any]) -> str:
        """Apply a single transformation to text."""
        pattern = transform["pattern"]
        replacement = transform["replacement"]
        count = transform["count"]
        
        # Apply the substitution
        result = pattern.sub(replacement, text, count=count)
        
        return result
    
    def _parse_flags(self, flag_names: List[str]) -> int:
        """Parse flag names to regex flags."""
        flags = 0
        flag_map = {
            "IGNORECASE": re.IGNORECASE,
            "MULTILINE": re.MULTILINE,
            "DOTALL": re.DOTALL,
            "VERBOSE": re.VERBOSE,
            "ASCII": re.ASCII,
            "LOCALE": re.LOCALE,
            "UNICODE": re.UNICODE,
            "I": re.IGNORECASE,
            "M": re.MULTILINE,
            "S": re.DOTALL,
            "X": re.VERBOSE,
            "A": re.ASCII,
            "L": re.LOCALE,
            "U": re.UNICODE,
        }
        
        for flag_name in flag_names:
            if flag_name in flag_map:
                flags |= flag_map[flag_name]
            else:
                LOG.warning(f"Unknown regex flag: {flag_name}")
        
        return flags
    
    def get_required_columns(self) -> List[str]:
        """Return fields that will be transformed."""
        return [t["field"] for t in self.transformations]


# Register the processor
register_processor("regex_transform", RegexTransformProcessor)