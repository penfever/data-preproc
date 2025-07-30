"""Filter-only strategy that preserves original dataset structure."""

from typing import Dict, Any, Optional
import logging

LOG = logging.getLogger(__name__)


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    """Load filter-only strategy that preserves original structure.
    
    This strategy doesn't do any tokenization or format conversion,
    it just returns the data as-is. Useful when you only want filtering
    without any transformations.
    """
    
    def filter_only_process(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Pass through examples unchanged - filtering is handled by processors."""
        LOG.debug(f"Filter-only processing {len(examples.get(list(examples.keys())[0], []))} examples")
        return examples
    
    return filter_only_process