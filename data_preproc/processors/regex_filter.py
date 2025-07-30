"""Regex-based filtering processor for pattern matching and content filtering."""

import logging
import re
from typing import Dict, Any, List, Optional, Union

from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)


class RegexFilterProcessor(DatasetProcessor):
    """Processor that filters examples based on regex pattern matching.
    
    Supports:
    - Multiple regex patterns per field
    - Include/exclude filtering logic (keep vs remove)
    - Case-sensitive and case-insensitive matching
    - Boolean logic for combining multiple patterns
    - Multiple fields in a single processor
    - Comprehensive logging of filter decisions
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Main configuration
        self.filter_patterns = config.get("filter_patterns", [])
        self.default_flags = self._parse_flags(config.get("default_flags", ["MULTILINE"]))
        self.logic_mode = config.get("logic_mode", "any").lower()  # "any" or "all"
        self.invert_logic = config.get("invert_logic", False)
        
        # Validation
        if not self.filter_patterns:
            raise ValueError("regex_filter requires at least one filter_pattern")
        
        if self.logic_mode not in ["any", "all"]:
            raise ValueError("logic_mode must be 'any' or 'all'")
        
        # Initialize statistics tracking first
        self.stats = {
            "total_processed": 0,
            "filtered_out": 0,
            "kept": 0,
            "pattern_matches": {}
        }
        
        # Compile patterns
        self.compiled_patterns = []
        for i, pattern_config in enumerate(self.filter_patterns):
            try:
                compiled_pattern = self._compile_pattern(pattern_config, i)
                self.compiled_patterns.append(compiled_pattern)
            except Exception as e:
                LOG.error(f"Error compiling filter pattern {i}: {e}")
                raise ValueError(f"Invalid filter pattern {i}: {e}")
        
        LOG.info(f"Initialized RegexFilterProcessor with {len(self.compiled_patterns)} patterns")
        LOG.info(f"Logic mode: {self.logic_mode}, Invert: {self.invert_logic}")
    
    def _compile_pattern(self, pattern_config: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Compile a single filter pattern configuration."""
        # Required fields
        field = pattern_config.get("field")
        if not field:
            raise ValueError(f"Pattern {index} must specify a 'field'")
        
        pattern = pattern_config.get("pattern")
        if not pattern:
            raise ValueError(f"Pattern {index} must specify a 'pattern'")
        
        # Optional fields with defaults
        action = pattern_config.get("action", "remove").lower()
        if action not in ["remove", "keep", "keep_only"]:
            raise ValueError(f"Pattern {index} action must be 'remove', 'keep', or 'keep_only'")
        
        description = pattern_config.get("description", f"Pattern {index}")
        flags = self._parse_flags(pattern_config.get("flags", []))
        
        # Combine default flags with pattern-specific flags
        combined_flags = self.default_flags | flags
        
        # Compile the regex pattern
        try:
            compiled_pattern = re.compile(pattern, combined_flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        compiled_config = {
            "field": field,
            "pattern": compiled_pattern,
            "original_pattern": pattern,
            "action": action,
            "description": description,
            "index": index
        }
        
        # Initialize stats for this pattern
        self.stats["pattern_matches"][index] = {
            "description": description,
            "matches": 0,
            "filtered": 0
        }
        
        return compiled_config
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter example based on regex patterns."""
        self.stats["total_processed"] += 1
        
        # Evaluate all patterns
        pattern_results = []
        
        for pattern_config in self.compiled_patterns:
            field = pattern_config["field"]
            pattern = pattern_config["pattern"]
            action = pattern_config["action"]
            index = pattern_config["index"]
            description = pattern_config["description"]
            
            # Check if field exists
            if field not in example:
                LOG.debug(f"Field '{field}' not found in example for pattern {index}")
                pattern_results.append(False)
                continue
            
            field_value = example[field]
            if not isinstance(field_value, str):
                LOG.debug(f"Field '{field}' is not a string for pattern {index}")
                pattern_results.append(False)
                continue
            
            # Test the pattern
            match = pattern.search(field_value) is not None
            
            if match:
                self.stats["pattern_matches"][index]["matches"] += 1
                LOG.debug(f"Pattern {index} ({description}) matched in field '{field}'")
            
            # Determine pattern result based on action
            if action == "remove" and match:
                pattern_results.append(True)  # True means "filter out"
            elif action in ["keep", "keep_only"] and match:
                pattern_results.append(False)  # False means "keep"
            elif action in ["keep", "keep_only"] and not match:
                pattern_results.append(True)  # True means "filter out" (didn't match keep condition)
            else:  # action == "remove" and not match
                pattern_results.append(False)  # False means "keep"
        
        # Apply boolean logic
        if self.logic_mode == "any":
            should_filter = any(pattern_results)
        else:  # "all"
            should_filter = all(pattern_results)
        
        # Apply inversion if specified
        if self.invert_logic:
            should_filter = not should_filter
        
        # Make filtering decision
        if should_filter:
            self.stats["filtered_out"] += 1
            
            # Update pattern-specific filter counts
            for i, (pattern_config, result) in enumerate(zip(self.compiled_patterns, pattern_results)):
                if result:  # This pattern contributed to filtering
                    self.stats["pattern_matches"][i]["filtered"] += 1
            
            LOG.debug(f"Example filtered out by regex patterns")
            return None
        else:
            self.stats["kept"] += 1
            return example
    
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
        """Return fields that will be checked by patterns."""
        return list(set(pattern["field"] for pattern in self.filter_patterns))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        if self.stats["total_processed"] > 0:
            filter_rate = self.stats["filtered_out"] / self.stats["total_processed"]
        else:
            filter_rate = 0.0
        
        return {
            **self.stats,
            "filter_rate": filter_rate,
            "keep_rate": 1.0 - filter_rate
        }
    
    def log_stats(self):
        """Log filtering statistics."""
        stats = self.get_stats()
        
        LOG.info(f"RegexFilterProcessor Statistics:")
        LOG.info(f"  Total processed: {stats['total_processed']}")
        LOG.info(f"  Filtered out: {stats['filtered_out']} ({stats['filter_rate']:.1%})")
        LOG.info(f"  Kept: {stats['kept']} ({stats['keep_rate']:.1%})")
        
        LOG.info(f"  Pattern-specific stats:")
        for pattern_stats in stats["pattern_matches"].values():
            desc = pattern_stats["description"]
            matches = pattern_stats["matches"]
            filtered = pattern_stats["filtered"]
            LOG.info(f"    {desc}: {matches} matches, {filtered} filtered")


# Register the processor
register_processor("regex_filter", RegexFilterProcessor)