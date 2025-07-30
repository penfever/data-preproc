"""Image toxicity filter processor using OpenCLIP (following LAION's approach)."""

from typing import Dict, Any, Optional, List, Tuple
import logging
import torch
import numpy as np
from . import DatasetProcessor, register_processor
from ..utils import ComputeDeviceUtils

LOG = logging.getLogger(__name__)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    LOG.warning("PIL not available for image processing")

# Lazy imports for OpenCLIP
_clip_model = None
_clip_preprocess = None
_tokenizer = None


def get_clip_model(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    """Lazily load and cache OpenCLIP model."""
    global _clip_model, _clip_preprocess, _tokenizer
    
    if _clip_model is None:
        try:
            import open_clip
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            _tokenizer = open_clip.get_tokenizer(model_name)
            _clip_model.eval()
            
            # Move to optimal device
            device = ComputeDeviceUtils.default_device()
            _clip_model = ComputeDeviceUtils.move_to_device(_clip_model, device)
            
            LOG.info(f"Loaded OpenCLIP model: {model_name} (pretrained: {pretrained}) on {device}")
        except ImportError:
            raise ImportError(
                "open-clip-torch is not installed. Please install with: pip install open-clip-torch"
            )
    
    return _clip_model, _clip_preprocess, _tokenizer


class ImageToxicityFilter(DatasetProcessor):
    """Filter images based on CLIP-based toxicity detection (LAION approach).
    
    This implements steps 6-9 from LAION-400M filtering:
    1. Compute CLIP embeddings for safety categories
    2. Calculate cosine similarities with image embeddings
    3. Tag images as NSFW/UNSURE/UNLIKELY based on similarities
    4. Additional filtering for underage-related content
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # CLIP model configuration
        self.model_name = config.get("model_name", "ViT-B-32")
        self.pretrained = config.get("pretrained", "openai")
        
        # Safety categories (following LAION)
        self.neutral_categories = config.get("neutral_categories", [
            "a photo", "a picture", "a drawing", "an illustration",
            "a painting", "a rendering", "a screenshot", "a selfie",
            "a landscape", "a portrait", "nature", "architecture"
        ])
        
        self.nsfw_categories = config.get("nsfw_categories", [
            "porn", "pornography", "sex", "sexual content", "nudity",
            "naked", "explicit content", "adult content", "nsfw content",
            "erotic", "xxx", "hardcore"
        ])
        
        # Additional categories for underage filtering
        self.underage_categories = config.get("underage_categories", [
            "child", "children", "kid", "kids", "minor", "minors",
            "teen", "teenager", "adolescent", "young", "youth",
            "baby", "infant", "toddler", "preteen", "underage"
        ])
        
        # Similarity thresholds
        self.nsfw_threshold = config.get("nsfw_threshold", 0.3)  # Conservative default
        self.underage_threshold = config.get("underage_threshold", 0.3)
        
        # Filtering strictness
        self.filter_nsfw = config.get("filter_nsfw", True)
        self.filter_unsure = config.get("filter_unsure", True)  # Conservative: also filter UNSURE
        self.filter_underage_risk = config.get("filter_underage_risk", True)
        
        # Whether to log filtered examples
        self.log_filtered = config.get("log_filtered", False)
        
        # Image field handling
        self.image_fields = config.get("image_fields", ["image", "images"])
        
        # Initialize statistics
        self.stats = {
            "total_processed": 0,
            "filtered": 0,
            "passed": 0,
            "no_images": 0,
            "nsfw_filtered": 0,
            "unsure_filtered": 0,
            "underage_risk_filtered": 0,
            "processing_errors": 0
        }
        
        # Models will be loaded on first use
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._category_embeddings = None
        
    def _get_category_embeddings(self):
        """Compute and cache category embeddings."""
        if self._category_embeddings is not None:
            return self._category_embeddings
            
        if self._model is None:
            self._model, self._preprocess, self._tokenizer = get_clip_model(
                self.model_name, self.pretrained
            )
        
        # Compute embeddings for all categories
        all_categories = (
            self.neutral_categories + 
            self.nsfw_categories + 
            self.underage_categories
        )
        
        # Tokenize and encode text
        with torch.no_grad():
            tokens = self._tokenizer(all_categories)
            device = ComputeDeviceUtils.default_device()
            tokens = ComputeDeviceUtils.move_to_device(tokens, device)
            
            text_embeddings = self._model.encode_text(tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        
        # Store embeddings by category type
        n_neutral = len(self.neutral_categories)
        n_nsfw = len(self.nsfw_categories)
        
        self._category_embeddings = {
            "neutral": text_embeddings[:n_neutral],
            "nsfw": text_embeddings[n_neutral:n_neutral + n_nsfw],
            "underage": text_embeddings[n_neutral + n_nsfw:],
            "all": text_embeddings
        }
        
        return self._category_embeddings
    
    def _process_image(self, image) -> Tuple[Optional[str], Optional[torch.Tensor]]:
        """Process image and return safety tag and embedding."""
        try:
            # Initialize models if needed
            if self._model is None:
                self._model, self._preprocess, self._tokenizer = get_clip_model(
                    self.model_name, self.pretrained
                )
            
            # Convert to PIL Image if necessary
            if not isinstance(image, Image.Image):
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                else:
                    return None, None
            
            # Preprocess image
            image_tensor = self._preprocess(image).unsqueeze(0)
            device = ComputeDeviceUtils.default_device()
            image_tensor = ComputeDeviceUtils.move_to_device(image_tensor, device)
            
            # Get image embedding
            with torch.no_grad():
                image_embedding = self._model.encode_image(image_tensor)
                image_embedding = image_embedding / image_embedding.norm(dim=1, keepdim=True)
            
            # Get category embeddings
            category_embeddings = self._get_category_embeddings()
            
            # Compute similarities with all categories
            similarities = torch.matmul(image_embedding, category_embeddings["all"].T).squeeze()
            
            # Split similarities by category type
            n_neutral = len(self.neutral_categories)
            n_nsfw = len(self.nsfw_categories)
            
            neutral_sims = similarities[:n_neutral]
            nsfw_sims = similarities[n_neutral:n_neutral + n_nsfw]
            underage_sims = similarities[n_neutral + n_nsfw:]
            
            # Find top similarities
            top_sims, top_indices = torch.topk(similarities, k=min(5, len(similarities)))
            
            # Determine safety tag (following LAION logic)
            nsfw_count = 0
            for idx in top_indices[:2]:  # Check top 2 similarities
                if idx >= n_neutral and idx < n_neutral + n_nsfw:
                    nsfw_count += 1
            
            if nsfw_count == 2:
                safety_tag = "NSFW"
            elif nsfw_count == 1:
                safety_tag = "UNSURE"
            else:
                safety_tag = "UNLIKELY"
            
            # Check for underage risk if image is NSFW or UNSURE
            if safety_tag in ["NSFW", "UNSURE"] and self.filter_underage_risk and len(underage_sims) > 0:
                max_underage_sim = torch.max(underage_sims).item()
                if max_underage_sim > self.underage_threshold:
                    safety_tag = "UNDERAGE_RISK"
            
            return safety_tag, image_embedding
            
        except Exception as e:
            LOG.warning(f"Error processing image: {e}")
            self.stats["processing_errors"] += 1
            return None, None
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single example, filtering based on image toxicity."""
        self.stats["total_processed"] += 1
        
        # Initialize models if needed
        if self._model is None:
            self._model, self._preprocess, self._tokenizer = get_clip_model(
                self.model_name, self.pretrained
            )
        
        # Extract images from example
        images = []
        for field in self.image_fields:
            if field in example and example[field] is not None:
                if isinstance(example[field], list):
                    images.extend(example[field])
                else:
                    images.append(example[field])
        
        if not images:
            self.stats["no_images"] += 1
            # No images to check, pass through
            self.stats["passed"] += 1
            return example
        
        # Check each image
        should_filter = False
        filter_reasons = []
        
        for idx, image in enumerate(images):
            safety_tag, _ = self._process_image(image)
            
            if safety_tag is None:
                continue
            
            # Apply filtering based on safety tag
            if safety_tag == "NSFW" and self.filter_nsfw:
                should_filter = True
                filter_reasons.append(f"Image {idx}: NSFW")
                self.stats["nsfw_filtered"] += 1
            elif safety_tag == "UNSURE" and self.filter_unsure:
                should_filter = True
                filter_reasons.append(f"Image {idx}: UNSURE")
                self.stats["unsure_filtered"] += 1
            elif safety_tag == "UNDERAGE_RISK":
                should_filter = True
                filter_reasons.append(f"Image {idx}: UNDERAGE_RISK")
                self.stats["underage_risk_filtered"] += 1
        
        if should_filter:
            self.stats["filtered"] += 1
            if self.log_filtered:
                LOG.info(f"Filtered example: {'; '.join(filter_reasons)}")
            return None
        
        self.stats["passed"] += 1
        return example
    
    def apply_to_dataset(self, dataset):
        """Apply filtering directly to the HF dataset for efficiency."""
        initial_count = len(dataset)
        LOG.info(f"🔍 Image Toxicity Filter: Processing {initial_count} examples")
        LOG.info(f"  Model: {self.model_name} (pretrained: {self.pretrained})")
        LOG.info(f"  Filter NSFW: {self.filter_nsfw}")
        LOG.info(f"  Filter UNSURE: {self.filter_unsure}")
        LOG.info(f"  Filter underage risk: {self.filter_underage_risk}")
        LOG.info(f"  Thresholds - NSFW: {self.nsfw_threshold}, Underage: {self.underage_threshold}")
        
        # Log device info
        ComputeDeviceUtils.log_device_info()
        
        # Reset statistics
        self.stats = {
            "total_processed": 0,
            "filtered": 0,
            "passed": 0,
            "no_images": 0,
            "nsfw_filtered": 0,
            "unsure_filtered": 0,
            "underage_risk_filtered": 0,
            "processing_errors": 0
        }
        
        # Apply filter
        filtered_dataset = dataset.filter(
            lambda example: self.process_example(example) is not None
        )
        
        final_count = len(filtered_dataset)
        filtered_count = initial_count - final_count
        
        # Log statistics
        LOG.info(f"📊 Image Toxicity Filter Results:")
        LOG.info(f"  ✅ Passed: {self.stats['passed']}/{initial_count} ({self.stats['passed']/initial_count*100:.1f}%)")
        LOG.info(f"  ❌ Filtered: {filtered_count}/{initial_count} ({filtered_count/initial_count*100:.1f}%)")
        LOG.info(f"  ⚠️  No images found: {self.stats['no_images']}")
        LOG.info(f"  🚫 NSFW filtered: {self.stats['nsfw_filtered']}")
        LOG.info(f"  ❓ UNSURE filtered: {self.stats['unsure_filtered']}")
        LOG.info(f"  👶 Underage risk filtered: {self.stats['underage_risk_filtered']}")
        
        if self.stats["processing_errors"] > 0:
            LOG.warning(f"  ⚠️  Processing errors: {self.stats['processing_errors']}")
        
        return filtered_dataset
    
    def get_required_columns(self) -> List[str]:
        """No hard requirements since we check multiple image fields."""
        return []


# Register the processor
register_processor("image_toxicity_filter", ImageToxicityFilter)