"""Error handling utilities for dataset processing."""

import logging
from typing import Optional, Dict, Any, Iterator, Callable
from dataclasses import dataclass
from enum import Enum

LOG = logging.getLogger(__name__)


class ErrorHandlingMode(Enum):
    """Error handling modes for dataset processing."""
    RAISE = "raise"
    LOG_AND_CONTINUE = "log_and_continue"
    RAISE_AFTER_K = "raise_after_k"


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling during dataset processing."""
    mode: ErrorHandlingMode = ErrorHandlingMode.RAISE
    max_errors: int = 10  # For RAISE_AFTER_K mode
    log_errors: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ErrorHandlingConfig":
        """Create ErrorHandlingConfig from dictionary."""
        mode_str = config.get("mode", "raise")
        try:
            mode = ErrorHandlingMode(mode_str)
        except ValueError:
            LOG.warning(f"Invalid error handling mode: {mode_str}. Using 'raise' mode.")
            mode = ErrorHandlingMode.RAISE
        
        return cls(
            mode=mode,
            max_errors=config.get("max_errors", 10),
            log_errors=config.get("log_errors", True)
        )


class ErrorHandlingWrapper:
    """Wrapper to handle errors during dataset iteration and processing."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.error_count = 0
        self.error_log = []
        
    def reset(self):
        """Reset error counts and logs."""
        self.error_count = 0
        self.error_log = []
        
    def handle_error(self, error: Exception, context: str = "", example_idx: Optional[int] = None) -> bool:
        """
        Handle an error based on the configured mode.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            example_idx: Index of the example that caused the error (if applicable)
            
        Returns:
            bool: True if processing should continue, False if it should stop
        """
        self.error_count += 1
        
        # Create error message with context
        error_msg = f"Error {self.error_count}"
        if example_idx is not None:
            error_msg += f" at example {example_idx}"
        if context:
            error_msg += f" in {context}"
        error_msg += f": {type(error).__name__}: {error}"
        
        # Log the error if enabled
        if self.config.log_errors:
            LOG.warning(error_msg)
            
        # Store error details
        self.error_log.append({
            "error": error,
            "context": context,
            "example_idx": example_idx,
            "error_type": type(error).__name__,
            "error_msg": str(error)
        })
        
        # Handle based on mode
        if self.config.mode == ErrorHandlingMode.RAISE:
            # Always raise the first error
            raise error
        elif self.config.mode == ErrorHandlingMode.LOG_AND_CONTINUE:
            # Log and continue processing
            return True
        elif self.config.mode == ErrorHandlingMode.RAISE_AFTER_K:
            # Raise after K errors
            if self.error_count > self.config.max_errors:
                LOG.error(f"Maximum error count ({self.config.max_errors}) exceeded. Stopping processing.")
                LOG.error(f"Total errors encountered: {self.error_count}")
                self._log_error_summary()
                raise error
            return True
        
        return False
    
    def _log_error_summary(self):
        """Log a summary of all errors encountered."""
        if not self.error_log:
            return
            
        error_types = {}
        for error_info in self.error_log:
            error_type = error_info["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        LOG.error("Error summary:")
        for error_type, count in error_types.items():
            LOG.error(f"  {error_type}: {count} occurrences")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of errors encountered."""
        error_types = {}
        for error_info in self.error_log:
            error_type = error_info["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        return {
            "total_errors": self.error_count,
            "error_types": error_types,
            "error_details": self.error_log
        }


def safe_dataset_iterator(dataset, error_handler: ErrorHandlingWrapper, context: str = "dataset iteration") -> Iterator[Any]:
    """
    Safely iterate over a dataset with error handling.
    
    Args:
        dataset: The dataset to iterate over
        error_handler: Error handling configuration
        context: Context string for error messages
        
    Yields:
        Examples from the dataset, skipping those that cause errors based on error handling mode
    """
    for i, example in enumerate(dataset):
        try:
            yield example
        except Exception as e:
            should_continue = error_handler.handle_error(e, context, i)
            if not should_continue:
                break


def safe_process_example(
    example: Any, 
    process_func: Callable[[Any], Any], 
    error_handler: ErrorHandlingWrapper,
    context: str = "example processing",
    example_idx: Optional[int] = None
) -> Optional[Any]:
    """
    Safely process an example with error handling.
    
    Args:
        example: The example to process
        process_func: Function to apply to the example
        error_handler: Error handling configuration
        context: Context string for error messages
        example_idx: Index of the example being processed
        
    Returns:
        Processed example or None if processing failed and should be skipped
    """
    try:
        return process_func(example)
    except Exception as e:
        should_continue = error_handler.handle_error(e, context, example_idx)
        if should_continue:
            return None  # Skip this example
        else:
            # Error handler decided to stop processing
            return None