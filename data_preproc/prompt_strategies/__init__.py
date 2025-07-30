"""Module to load prompt strategies."""

import importlib
import inspect

from data_preproc.utils.logging import get_logger

LOG = get_logger(__name__)


def load(strategy, tokenizer, cfg, ds_cfg=None, processor=None):
    """
    Load a prompt strategy dynamically.

    Args:
        strategy: Strategy name or module path
        tokenizer: Tokenizer instance
        cfg: Main configuration
        ds_cfg: Dataset-specific configuration
        processor: Optional processor for multimodal models

    Returns:
        Strategy function or None if not found
    """
    try:
        # Handle built-in strategies
        if strategy == "alpaca":
            from .alpaca import load as alpaca_load
            return alpaca_load(tokenizer, cfg, ds_cfg)
        elif strategy == "completion":
            from .completion import load as completion_load
            return completion_load(tokenizer, cfg, ds_cfg)
        elif strategy == "chat_template":
            from .chat_template import load as chat_template_load
            return chat_template_load(tokenizer, cfg, ds_cfg)
        elif strategy == "vision_language" or strategy == "vl":
            from .vision_language import load as vision_language_load
            return vision_language_load(tokenizer, cfg, ds_cfg)
        
        # Try to load custom strategy
        load_fn = "load"
        package = "data_preproc.prompt_strategies"
        
        # Check if a specific load function is specified
        if strategy.split(".")[-1].startswith("load_"):
            load_fn = strategy.split(".")[-1]
            strategy = ".".join(strategy.split(".")[:-1])
        
        # Import the module
        mod = importlib.import_module(f".{strategy}", package)
        func = getattr(mod, load_fn)
        
        # Prepare kwargs based on function signature
        load_kwargs = {}
        sig = inspect.signature(func)
        if "ds_cfg" in sig.parameters:
            load_kwargs["ds_cfg"] = ds_cfg
        if "processor" in sig.parameters:
            load_kwargs["processor"] = processor
        
        return func(tokenizer, cfg, **load_kwargs)
        
    except ModuleNotFoundError:
        LOG.warning(f"Strategy '{strategy}' not found")
        return None
    except Exception as exc:
        LOG.error(f"Failed to load prompt strategy '{strategy}': {str(exc)}")
        raise exc