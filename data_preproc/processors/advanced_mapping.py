"""Advanced mapping processor for complex schema transformations."""

import logging
import gc
from typing import Dict, Any, List, Optional, Union
from copy import deepcopy

from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)


class AdvancedMappingProcessor(DatasetProcessor):
    """Processor that supports advanced field mapping with nested extraction.
    
    Supports:
    - Nested field extraction using dot notation
    - List indexing and filtering
    - Multiple target fields from single source
    - Conditional mappings based on field values
    - Fallback field extraction
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mappings = config.get("mappings", [])
        self.simple_mappings = config.get("simple_mappings", {})
        self.keep_unmapped = config.get("keep_unmapped", True)
        self.remove_source_fields = config.get("remove_source_fields", False)
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply advanced mappings to transform example."""
        try:
            # Start with empty result to save memory if keep_unmapped is False
            if self.keep_unmapped:
                # Use shallow copy for better memory efficiency
                result = example.copy()
            else:
                result = {}
            
            LOG.debug(f"Advanced mapping processing example with keys: {list(example.keys())}")
            
            # Apply simple mappings first (backward compatibility)
            for old_name, new_name in self.simple_mappings.items():
                if old_name in example:
                    result[new_name] = example[old_name]
                    LOG.debug(f"Simple mapping: {old_name} -> {new_name}")
            
            # Apply advanced mappings
            for i, mapping in enumerate(self.mappings):
                source = mapping.get("source")
                if not source:
                    LOG.debug(f"Mapping {i}: No source field specified")
                    continue
                
                LOG.debug(f"Mapping {i}: Processing source '{source}'")
                
                # Remove fields before processing if specified
                remove_before = mapping.get("remove_before", [])
                if isinstance(remove_before, str):
                    remove_before = [remove_before]
                for field in remove_before:
                    if field in result:
                        result.pop(field, None)
                        LOG.debug(f"Mapping {i}: Removed field before processing: {field}")
                
                # Extract value from source - use example for source data extraction
                # Only use result for checking already processed fields
                value = self._extract_value(example, mapping)
                
                if value is not None:
                    # Apply to target(s)
                    targets = mapping.get("targets", [])
                    target = mapping.get("target")
                    
                    # Handle both single target and multiple targets
                    if target:
                        targets = [target]
                    
                    for target_field in targets:
                        result[target_field] = value
                        LOG.debug(f"Mapping {i}: Set {target_field} = {type(value).__name__}")
                else:
                    LOG.debug(f"Mapping {i}: Extracted value is None for source '{source}'")
                
                # Remove fields after processing if specified
                remove_after = mapping.get("remove_after", [])
                if isinstance(remove_after, str):
                    remove_after = [remove_after]
                for field in remove_after:
                    if field in result:
                        result.pop(field, None)
                        LOG.debug(f"Mapping {i}: Removed field after processing: {field}")
            
            # Validate that we have minimum required fields
            required_fields = ["image", "problem", "solution"]
            missing_fields = [f for f in required_fields if f not in result]
            if missing_fields:
                LOG.debug(f"Filtering example due to missing required fields: {missing_fields}")
                LOG.debug(f"Available fields in result: {list(result.keys())}")
                return None
            
            LOG.debug(f"Advanced mapping successful, result keys: {list(result.keys())}")
            return result
            
        except Exception as e:
            LOG.warning(f"Error in advanced mapping: {e}")
            LOG.debug(f"Example keys that caused error: {list(example.keys()) if example else 'None'}")
            return example if self.keep_unmapped else None
    
    def _extract_value(self, example: Dict[str, Any], mapping: Dict[str, Any]) -> Any:
        """Extract value from example based on mapping configuration."""
        source = mapping.get("source", "")
        
        # Get the base value
        value = self._navigate_path(example, source)
        
        # Apply filters if this is a list
        if isinstance(value, list) and "filter" in mapping:
            value = self._filter_list(value, mapping["filter"])
        
        # Extract specific field from result
        if "extract" in mapping:
            if isinstance(value, list):
                # Extract field from each item in list
                extracted = []
                for item in value:
                    if isinstance(item, dict):
                        extracted.append(self._navigate_path(item, mapping["extract"]))
                value = extracted
            elif isinstance(value, dict):
                value = self._navigate_path(value, mapping["extract"])
        
        # Handle extract_first_of - try multiple fields in order
        if "extract_first_of" in mapping:
            if isinstance(value, dict):
                for field in mapping["extract_first_of"]:
                    extracted = self._navigate_path(value, field)
                    if extracted is not None:
                        value = extracted
                        break
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # For lists, extract from first item
                for field in mapping["extract_first_of"]:
                    extracted = self._navigate_path(value[0], field)
                    if extracted is not None:
                        value = extracted
                        break
        
        # If we have a list but need a single value, take the first
        if isinstance(value, list) and mapping.get("take_first", True):
            if len(value) > 0:
                value = value[0]
            else:
                value = None
        
        return value
    
    def _navigate_path(self, obj: Any, path: str) -> Any:
        """Navigate a dot-separated path in an object.
        
        Supports:
        - Dot notation: "field.subfield"
        - List indexing: "field[0]"
        - Negative indexing: "field[-1]"
        """
        if not path:
            return obj
        
        current = obj
        parts = path.split(".")
        
        for part in parts:
            if current is None:
                return None
            
            # Check for list indexing
            if "[" in part and "]" in part:
                field_name, index_part = part.split("[", 1)
                index_str = index_part.rstrip("]")
                
                # Navigate to the field first if there is one
                if field_name:
                    if isinstance(current, dict):
                        current = current.get(field_name)
                    else:
                        return None
                
                # Apply index
                if isinstance(current, list):
                    try:
                        index = int(index_str)
                        if -len(current) <= index < len(current):
                            current = current[index]
                        else:
                            return None
                    except (ValueError, IndexError):
                        return None
                else:
                    return None
            else:
                # Regular field access
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
        
        return current
    
    def _filter_list(self, lst: List[Any], filter_spec: Dict[str, Any]) -> List[Any]:
        """Filter a list based on field values."""
        if not isinstance(lst, list):
            return lst
        
        result = []
        for item in lst:
            if isinstance(item, dict):
                # Check all filter conditions
                match = True
                for field, expected_value in filter_spec.items():
                    if item.get(field) != expected_value:
                        match = False
                        break
                
                if match:
                    result.append(item)
            else:
                # Can't filter non-dict items
                result.append(item)
        
        return result
    
    def get_required_columns(self) -> List[str]:
        """Extract required top-level columns from mappings."""
        required = set()
        
        # From simple mappings
        required.update(self.simple_mappings.keys())
        
        # From advanced mappings - only top-level fields
        for mapping in self.mappings:
            source = mapping.get("source", "")
            if source:
                # Extract top-level field name
                top_field = source.split(".")[0].split("[")[0]
                if top_field:
                    required.add(top_field)
        
        return list(required)


# Register the processor
register_processor("advanced_mapping", AdvancedMappingProcessor)