"""Init for data_preproc.loaders module"""

from .processor import load_processor
from .tokenizer import load_tokenizer

__all__ = ["load_processor", "load_tokenizer"]