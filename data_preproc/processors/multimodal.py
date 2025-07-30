"""Multimodal dataset processors."""

from typing import Dict, Any, Optional, List
import logging
from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    LOG.warning("PIL not available for image processing")


class QAToMessagesProcessor(DatasetProcessor):
    """Convert Q&A format to messages format."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.question_field = config.get("question_field", "problem")
        self.answer_field = config.get("answer_field", "solution")
        self.image_field = config.get("image_field", "image")
        
        # Alternative field names to try
        self.question_alternatives = config.get("question_alternatives", 
                                               ["question", "instruction", "input", "original_question"])
        self.answer_alternatives = config.get("answer_alternatives", 
                                             ["answer", "output", "response", "original_answer"])
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Q&A format to messages format."""
        # Find question field
        question_content = None
        question_field_used = None
        
        for field in [self.question_field] + self.question_alternatives:
            if field in example and example[field]:
                question_content = example[field]
                question_field_used = field
                break
        
        # Find answer field  
        answer_content = None
        answer_field_used = None
        
        for field in [self.answer_field] + self.answer_alternatives:
            if field in example and example[field]:
                answer_content = example[field]
                answer_field_used = field
                break
        
        if not question_content or not answer_content:
            LOG.debug(f"Skipping example: missing question ({question_field_used}) or answer ({answer_field_used})")
            return None
        
        # Build messages
        messages = [
            {"role": "user", "content": str(question_content)},
            {"role": "assistant", "content": str(answer_content)}
        ]
        
        # Handle images
        images = []
        if self.image_field in example and example[self.image_field] is not None:
            images = [example[self.image_field]]
        elif "images" in example and example["images"] is not None:
            images = example["images"] if isinstance(example["images"], list) else [example["images"]]
        
        # Build converted example
        converted = {
            "messages": messages,
            "images": images,
            "videos": [],
            "audios": []
        }
        
        # Preserve other fields as metadata
        metadata = {}
        for key, value in example.items():
            if key not in [question_field_used, answer_field_used, self.image_field, "images"]:
                metadata[key] = value
        
        if metadata:
            converted["metadata"] = metadata
        
        return converted
    
    def get_required_columns(self) -> List[str]:
        """No hard requirements since we try alternatives."""
        return []


class MultimodalFilterProcessor(DatasetProcessor):
    """Filter multimodal data based on image/video/audio criteria."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.filter_corrupted_images = config.get("filter_corrupted_images", False)
        self.require_images = config.get("require_images", False)
        self.max_image_size = config.get("max_image_size")
        self.min_image_size = config.get("min_image_size")
        
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter based on multimodal criteria."""
        # Check image requirements
        if self.require_images:
            images = example.get("images", []) or example.get("image", [])
            if not images:
                LOG.debug("Filtering example: no images found but required")
                return None
        
        # Filter corrupted images
        if self.filter_corrupted_images:
            if "image" in example and example["image"]:
                if not self._validate_image(example["image"]):
                    LOG.debug("Filtering example: corrupted image")
                    return None
            
            if "images" in example and example["images"]:
                valid_images = []
                for img in example["images"]:
                    if self._validate_image(img):
                        valid_images.append(img)
                
                if len(valid_images) == 0 and len(example["images"]) > 0:
                    LOG.debug("Filtering example: all images corrupted")
                    return None
                
                # Update with valid images only
                example = dict(example)  # Copy
                example["images"] = valid_images
        
        return example
    
    def _validate_image(self, image) -> bool:
        """Validate image integrity and size."""
        if not HAS_PIL:
            return True  # Skip validation if PIL not available
        
        try:
            if hasattr(image, "verify"):
                # Make a copy since verify() can modify the image
                img_copy = image.copy()
                img_copy.verify()
            
            # Check size constraints
            if hasattr(image, "size"):
                width, height = image.size
                
                if self.max_image_size:
                    max_w, max_h = self.max_image_size
                    if width > max_w or height > max_h:
                        LOG.debug(f"Image too large: {width}x{height} > {max_w}x{max_h}")
                        return False
                
                if self.min_image_size:
                    min_w, min_h = self.min_image_size
                    if width < min_w or height < min_h:
                        LOG.debug(f"Image too small: {width}x{height} < {min_w}x{min_h}")
                        return False
            
            return True
            
        except Exception as e:
            LOG.debug(f"Image validation failed: {e}")
            return False
    
    def get_required_columns(self) -> List[str]:
        """No hard requirements."""
        return []


# Register multimodal processors
register_processor("qa_to_messages", QAToMessagesProcessor)
register_processor("multimodal_filter", MultimodalFilterProcessor)