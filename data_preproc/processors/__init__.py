"""Dataset processors for different formats."""

from typing import Dict, Type, Any, Optional, Union, List
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

LOG = logging.getLogger(__name__)

# Global registry for dataset processor classes
PROCESSORS: Dict[str, Type["DatasetProcessor"]] = {}

# Global registry for named processor instances
PROCESSOR_INSTANCES: Dict[str, "DatasetProcessor"] = {}


@dataclass 
class ProcessorCondition:
    """Conditions for conditional processor execution."""
    field_exists: Optional[List[str]] = None           # Fields that must exist
    field_not_exists: Optional[List[str]] = None       # Fields that must not exist
    field_equals: Optional[Dict[str, Any]] = None      # Field values that must match
    field_not_equals: Optional[Dict[str, Any]] = None  # Field values that must not match
    custom: Optional[str] = None                       # Custom condition expression
    
    def evaluate(self, example: Dict[str, Any]) -> bool:
        """Evaluate if conditions are met for the given example."""
        # Check field existence requirements
        if self.field_exists:
            if not all(field in example for field in self.field_exists):
                return False
                
        if self.field_not_exists:
            if any(field in example for field in self.field_not_exists):
                return False
                
        # Check field value requirements
        if self.field_equals:
            for field, expected_value in self.field_equals.items():
                if field not in example or example[field] != expected_value:
                    return False
                    
        if self.field_not_equals:
            for field, forbidden_value in self.field_not_equals.items():
                if field in example and example[field] == forbidden_value:
                    return False
        
        # Custom condition evaluation could be added here
        if self.custom:
            # For now, just log that custom conditions aren't implemented
            LOG.warning(f"Custom condition '{self.custom}' not yet implemented, skipping")
            
        return True


@dataclass
class ProcessorConfig:
    """Enhanced processor configuration with modular features."""
    type: str                                          # Processor type (required)
    name: Optional[str] = None                         # Optional unique name
    condition: Optional[ProcessorCondition] = None     # Conditional execution
    reference: Optional[str] = None                    # Reference to existing named processor
    parameters: Dict[str, Any] = field(default_factory=dict)  # Processor-specific config
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ProcessorConfig":
        """Create ProcessorConfig from dictionary (YAML config)."""
        # Handle backward compatibility - if no 'type' at top level, assume entire dict is parameters
        if 'type' not in config:
            raise ValueError("Processor configuration must have 'type' field")
            
        processor_type = config.pop('type')
        name = config.pop('name', None)
        reference = config.pop('reference', None)
        
        # Handle condition parsing
        condition = None
        if 'condition' in config:
            condition_dict = config.pop('condition')
            condition = ProcessorCondition(**condition_dict)
            
        # Everything else becomes parameters
        parameters = config
        
        return cls(
            type=processor_type,
            name=name,
            condition=condition, 
            reference=reference,
            parameters=parameters
        )


class DatasetProcessor(ABC):
    """Abstract base class for dataset processors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize processor with configuration."""
        self.config = config
        self.name = config.get('name')  # Store optional name
        
        # Parse condition if provided
        self.condition = None
        if 'condition' in config:
            condition_dict = config['condition']
            if isinstance(condition_dict, dict):
                self.condition = ProcessorCondition(**condition_dict)
            elif isinstance(condition_dict, ProcessorCondition):
                self.condition = condition_dict
    
    @abstractmethod
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single example.
        
        Args:
            example: Input example dictionary
            
        Returns:
            Processed example or None if example should be filtered
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """Get list of required columns for this processor."""
        pass
    
    def validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate that example has required columns."""
        required = self.get_required_columns()
        return all(col in example for col in required)
    
    def should_process(self, example: Dict[str, Any]) -> bool:
        """Check if this processor should process the given example."""
        # First check if example is valid
        if not self.validate_example(example):
            return False
            
        # Then check conditional execution
        if self.condition:
            return self.condition.evaluate(example)
            
        return True


def register_processor(name: str, processor_class: Type[DatasetProcessor]):
    """Register a dataset processor class."""
    PROCESSORS[name] = processor_class
    LOG.debug(f"Registered processor class: {name}")


def register_processor_instance(name: str, processor_instance: DatasetProcessor):
    """Register a named processor instance for reuse."""
    PROCESSOR_INSTANCES[name] = processor_instance
    LOG.debug(f"Registered processor instance: {name}")


def get_processor_class(processor_type: str) -> Type[DatasetProcessor]:
    """Get a processor class by type."""
    if processor_type not in PROCESSORS:
        raise ValueError(f"Unknown processor type: {processor_type}. Available: {list(PROCESSORS.keys())}")
    return PROCESSORS[processor_type]


def get_processor_instance(name: str) -> DatasetProcessor:
    """Get a registered processor instance by name."""
    if name not in PROCESSOR_INSTANCES:
        raise ValueError(f"Unknown processor instance: {name}. Available: {list(PROCESSOR_INSTANCES.keys())}")
    return PROCESSOR_INSTANCES[name]


def create_processor(config: Union[Dict[str, Any], ProcessorConfig]) -> DatasetProcessor:
    """Create a processor instance from configuration.
    
    Args:
        config: Processor configuration (dict from YAML or ProcessorConfig object)
        
    Returns:
        DatasetProcessor instance
    """
    # Convert dict to ProcessorConfig if needed
    if isinstance(config, dict):
        # Handle backward compatibility - if 'type' is at root level, use new format
        if 'type' in config:
            # Make a copy to avoid modifying original
            config_copy = config.copy()
            proc_config = ProcessorConfig.from_dict(config_copy)
        else:
            # Old format - assume entire dict is the config with implicit type
            raise ValueError("Processor configuration must specify 'type' field")
    else:
        proc_config = config
    
    # Handle processor references
    if proc_config.reference:
        LOG.debug(f"Using processor reference: {proc_config.reference}")
        return get_processor_instance(proc_config.reference)
    
    # Create new processor instance
    processor_class = get_processor_class(proc_config.type)
    
    # Combine parameters with metadata for backward compatibility
    full_config = proc_config.parameters.copy()
    if proc_config.name:
        full_config['name'] = proc_config.name
    if proc_config.condition:
        full_config['condition'] = proc_config.condition
        
    processor = processor_class(full_config)
    
    # Register named instance if name is provided
    if proc_config.name:
        register_processor_instance(proc_config.name, processor)
        
    return processor


def get_processor(name_or_config: Union[str, Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> DatasetProcessor:
    """Get a processor instance by name or create from configuration.
    
    This function maintains backward compatibility while supporting new features.
    
    Args:
        name_or_config: Either processor type name (old style) or full config dict (new style)
        config: Configuration dict (only used with old style)
        
    Returns:
        DatasetProcessor instance
    """
    # Handle backward compatibility
    if isinstance(name_or_config, str) and config is not None:
        # Old style: get_processor("processor_type", {...})
        LOG.debug(f"Using backward compatible processor creation for type: {name_or_config}")
        full_config = config.copy()
        full_config['type'] = name_or_config
        return create_processor(full_config)
    elif isinstance(name_or_config, dict):
        # New style: get_processor({...})
        return create_processor(name_or_config)
    elif isinstance(name_or_config, str):
        # Try to get as instance reference first, then as type
        try:
            return get_processor_instance(name_or_config)
        except ValueError:
            # Fall back to creating with empty config (for backward compatibility)
            return create_processor({'type': name_or_config})
    else:
        raise ValueError(f"Invalid processor specification: {name_or_config}")


def list_processors() -> list[str]:
    """List all registered processor types."""
    return list(PROCESSORS.keys())


def list_processor_instances() -> list[str]:
    """List all registered processor instances."""
    return list(PROCESSOR_INSTANCES.keys())


def clear_processor_instances():
    """Clear all registered processor instances (useful for testing)."""
    global PROCESSOR_INSTANCES
    PROCESSOR_INSTANCES.clear()
    LOG.debug("Cleared all processor instances")