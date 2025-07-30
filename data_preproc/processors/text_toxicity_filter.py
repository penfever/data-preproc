"""Text toxicity filter processor using Detoxify."""

from typing import Dict, Any, Optional, List
import logging
from . import DatasetProcessor, register_processor
from ..utils import ComputeDeviceUtils

LOG = logging.getLogger(__name__)

# Lazy import to avoid loading model at import time
_detoxify_model = None


def get_detoxify_model(model_type: str = "original"):
    """Lazily load and cache Detoxify model."""
    global _detoxify_model
    if _detoxify_model is None:
        try:
            from detoxify import Detoxify
            device = ComputeDeviceUtils.default_device()
            device_str = device.type if device.type != 'mps' else 'cpu'  # Detoxify may not support MPS
            _detoxify_model = Detoxify(model_type, device=device_str)
            LOG.info(f"Loaded Detoxify model: {model_type} on {device_str}")
        except ImportError:
            raise ImportError(
                "Detoxify is not installed. Please install with: pip install detoxify"
            )
    return _detoxify_model


class TextToxicityFilter(DatasetProcessor):
    """Filter text content based on toxicity scores using Detoxify.
    
    Detoxify provides scores for multiple toxicity types:
    - toxicity: Overall toxicity score
    - severe_toxicity: Severe toxicity score
    - obscene: Obscene content score
    - threat: Threatening content score
    - insult: Insulting content score
    - identity_attack: Identity-based attack score
    - sexual_explicit: Sexually explicit content score (multilingual model only)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model selection: 'original', 'unbiased', or 'multilingual'
        self.model_type = config.get("model_type", "original")
        
        # Text field extraction preferences (same as hf_filter)
        self.text_fields = config.get("text_fields", ["problem", "solution", "question", "answer", "text", "content"])
        
        # Toxicity thresholds (0.0 to 1.0, where 1.0 is most toxic)
        # Default thresholds are conservative to avoid false positives
        self.thresholds = {
            "toxicity": config.get("toxicity_threshold", 0.7),
            "severe_toxicity": config.get("severe_toxicity_threshold", 0.5),
            "obscene": config.get("obscene_threshold", 0.7),
            "threat": config.get("threat_threshold", 0.7),
            "insult": config.get("insult_threshold", 0.7),
            "identity_attack": config.get("identity_attack_threshold", 0.7),
            "sexual_explicit": config.get("sexual_explicit_threshold", 0.7),
        }
        
        # Which toxicity types to check (by default, check all)
        self.check_types = config.get("check_types", [
            "toxicity", "severe_toxicity", "obscene", 
            "threat", "insult", "identity_attack"
        ])
        
        # Filtering mode: 'any' (filter if ANY type exceeds threshold) or 'all' (filter if ALL types exceed)
        self.filter_mode = config.get("filter_mode", "any")
        
        # Whether to log examples that are filtered
        self.log_filtered = config.get("log_filtered", False)
        
        # Initialize statistics
        self.stats = {
            "total_processed": 0,
            "filtered": 0,
            "passed": 0,
            "no_text_content": 0,
            "filtered_by_type": {t: 0 for t in self.check_types}
        }
        
        # Model will be loaded on first use
        self._model = None
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single example, filtering based on toxicity."""
        self.stats["total_processed"] += 1
        
        # Extract text content
        text_content = self._extract_text_content(example)
        if not text_content:
            self.stats["no_text_content"] += 1
            # If no text content, pass through (don't filter)
            self.stats["passed"] += 1
            return example
        
        # Get toxicity scores
        try:
            if self._model is None:
                self._model = get_detoxify_model(self.model_type)
            
            scores = self._model.predict(text_content)
            
            # Check toxicity levels
            exceeds_threshold = []
            for toxicity_type in self.check_types:
                if toxicity_type not in scores:
                    continue
                    
                score = scores[toxicity_type]
                threshold = self.thresholds.get(toxicity_type, 0.7)
                
                if score > threshold:
                    exceeds_threshold.append(toxicity_type)
                    self.stats["filtered_by_type"][toxicity_type] += 1
            
            # Apply filtering logic
            should_filter = False
            if self.filter_mode == "any" and len(exceeds_threshold) > 0:
                should_filter = True
            elif self.filter_mode == "all" and len(exceeds_threshold) == len(self.check_types):
                should_filter = True
            
            if should_filter:
                self.stats["filtered"] += 1
                if self.log_filtered:
                    LOG.info(f"Filtered example with toxicity types: {exceeds_threshold}")
                    LOG.debug(f"Scores: {scores}")
                    LOG.debug(f"Text preview: {text_content[:200]}...")
                return None
            
            self.stats["passed"] += 1
            return example
            
        except Exception as e:
            LOG.warning(f"Error processing example for toxicity: {e}")
            # On error, pass through (don't filter)
            self.stats["passed"] += 1
            return example
    
    def _extract_text_content(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract text content from example for toxicity analysis."""
        texts = []
        
        for field in self.text_fields:
            if field in example and example[field]:
                texts.append(str(example[field]))
        
        return " ".join(texts) if texts else None
    
    def apply_to_dataset(self, dataset):
        """Apply filtering directly to the HF dataset for efficiency."""
        initial_count = len(dataset)
        LOG.info(f"🔍 Text Toxicity Filter: Processing {initial_count} examples")
        LOG.info(f"  Model: {self.model_type}")
        LOG.info(f"  Checking types: {self.check_types}")
        LOG.info(f"  Filter mode: {self.filter_mode}")
        LOG.info(f"  Thresholds: {self.thresholds}")
        
        # Log device info
        ComputeDeviceUtils.log_device_info()
        
        # Reset statistics
        self.stats = {
            "total_processed": 0,
            "filtered": 0,
            "passed": 0,
            "no_text_content": 0,
            "filtered_by_type": {t: 0 for t in self.check_types}
        }
        
        # Apply filter
        filtered_dataset = dataset.filter(
            lambda example: self.process_example(example) is not None
        )
        
        final_count = len(filtered_dataset)
        filtered_count = initial_count - final_count
        
        # Log statistics
        LOG.info(f"📊 Text Toxicity Filter Results:")
        LOG.info(f"  ✅ Passed: {self.stats['passed']}/{initial_count} ({self.stats['passed']/initial_count*100:.1f}%)")
        LOG.info(f"  ❌ Filtered: {filtered_count}/{initial_count} ({filtered_count/initial_count*100:.1f}%)")
        LOG.info(f"  ⚠️  No text content: {self.stats['no_text_content']}")
        
        if filtered_count > 0:
            LOG.info(f"  Filtered by toxicity type:")
            for toxicity_type, count in self.stats["filtered_by_type"].items():
                if count > 0:
                    LOG.info(f"    - {toxicity_type}: {count}")
        
        return filtered_dataset
    
    def get_required_columns(self) -> List[str]:
        """No hard requirements since we try multiple text fields."""
        return []


# Register the processor
register_processor("text_toxicity_filter", TextToxicityFilter)