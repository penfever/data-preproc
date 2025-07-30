"""Pipeline processor for composing multiple processors into reusable sequences."""

import logging
from typing import Dict, Any, Optional, List, Union
from datasets import Dataset

from . import DatasetProcessor, register_processor, get_processor_instance, create_processor

LOG = logging.getLogger(__name__)


class PipelineProcessor(DatasetProcessor):
    """A processor that composes multiple processors into a pipeline.
    
    This allows creating reusable processor sequences and enables modular composition.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.processors: List[DatasetProcessor] = []
        self.processor_configs = config.get("processors", [])
        
        if not self.processor_configs:
            raise ValueError("Pipeline processor requires 'processors' configuration")
        
        # Initialize processors from configuration
        self._initialize_processors()
        
    def _initialize_processors(self):
        """Initialize all processors in the pipeline."""
        for i, proc_config in enumerate(self.processor_configs):
            try:
                if isinstance(proc_config, str):
                    # Reference to existing named processor
                    processor = get_processor_instance(proc_config)
                    LOG.debug(f"Pipeline step {i+1}: Using processor reference '{proc_config}'")
                elif isinstance(proc_config, dict):
                    # Inline processor configuration
                    processor = create_processor(proc_config)
                    proc_name = proc_config.get("name", proc_config.get("type", f"step_{i+1}"))
                    LOG.debug(f"Pipeline step {i+1}: Created processor '{proc_name}'")
                else:
                    raise ValueError(f"Invalid processor configuration at step {i+1}: {proc_config}")
                
                self.processors.append(processor)
                
            except Exception as e:
                LOG.error(f"Failed to initialize processor at pipeline step {i+1}: {e}")
                raise
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process an example through the entire pipeline.
        
        Args:
            example: Input example dictionary
            
        Returns:
            Processed example or None if filtered by any processor
        """
        current_example = example
        
        for i, processor in enumerate(self.processors):
            if current_example is None:
                # Example was filtered by previous processor
                return None
            
            try:
                # Check if processor should process this example
                if hasattr(processor, 'should_process') and not processor.should_process(current_example):
                    # Skip this processor but continue with next one
                    LOG.debug(f"Pipeline step {i+1}: Skipping due to condition")
                    continue
                
                # Process the example
                current_example = processor.process_example(current_example)
                
                if current_example is None:
                    # Example was filtered
                    LOG.debug(f"Pipeline step {i+1}: Example filtered")
                    return None
                    
            except Exception as e:
                LOG.warning(f"Error in pipeline step {i+1}: {e}")
                # Set filter reason for debugging
                self._last_filter_reason = f"pipeline_step_{i+1}_error"
                return None
        
        return current_example
    
    def get_required_columns(self) -> list[str]:
        """Get list of required columns for this pipeline.
        
        Returns the union of required columns from all processors.
        """
        required_columns = set()
        
        for processor in self.processors:
            try:
                proc_columns = processor.get_required_columns()
                required_columns.update(proc_columns)
            except Exception as e:
                LOG.warning(f"Could not get required columns from processor: {e}")
        
        return list(required_columns)
    
    def apply_to_dataset(self, dataset: Dataset) -> Dataset:
        """Apply the entire pipeline to a dataset.
        
        This method allows pipeline processors to be treated as special
        dataset-level processors when needed.
        """
        current_dataset = dataset
        
        for i, processor in enumerate(self.processors):
            try:
                # Check if processor has special dataset-level method
                if hasattr(processor, 'apply_to_dataset'):
                    current_dataset = processor.apply_to_dataset(current_dataset)
                else:
                    # Apply processor example by example
                    processed_examples = []
                    
                    for example in current_dataset:
                        # Check conditions and process
                        if hasattr(processor, 'should_process') and not processor.should_process(example):
                            processed_examples.append(example)  # Keep original
                        else:
                            processed = processor.process_example(example)
                            if processed is not None:
                                processed_examples.append(processed)
                    
                    # Create new dataset from processed examples
                    if processed_examples:
                        current_dataset = Dataset.from_list(processed_examples)
                    else:
                        # All examples were filtered
                        LOG.warning(f"Pipeline step {i+1}: All examples filtered")
                        return Dataset.from_list([])
                
                LOG.info(f"Pipeline step {i+1}: {len(current_dataset)} examples remaining")
                
            except Exception as e:
                LOG.error(f"Error applying pipeline step {i+1} to dataset: {e}")
                raise
        
        return current_dataset
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        step_names = []
        for processor in self.processors:
            if hasattr(processor, 'name') and processor.name:
                step_names.append(processor.name)
            else:
                step_names.append(processor.__class__.__name__)
        
        pipeline_name = self.name or "Pipeline"
        return f"{pipeline_name}({' -> '.join(step_names)})"


# Register the pipeline processor
register_processor("pipeline", PipelineProcessor)