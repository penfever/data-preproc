"""Processor loading functionality for multimodal models"""

from transformers import AutoProcessor

from data_preproc.utils.logging import get_logger

LOG = get_logger(__name__)


def load_processor(cfg, tokenizer=None):
    """
    Load processor for multimodal models.

    Args:
        cfg: Configuration object
        tokenizer: Optional tokenizer to use with processor

    Returns:
        Loaded processor or None
    """
    if not cfg.processor_type:
        return None

    processor_path = cfg.processor_config or cfg.base_model

    try:
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=cfg.trust_remote_code or False,
        )

        # Set tokenizer if provided
        if tokenizer and hasattr(processor, "tokenizer"):
            processor.tokenizer = tokenizer

        LOG.info(f"Loaded processor from {processor_path}")
        return processor

    except Exception as e:
        LOG.error(f"Failed to load processor: {e}")
        return None