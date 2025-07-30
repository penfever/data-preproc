"""Filter dataset samples based on the number of images they contain."""

from typing import Dict, Any, Optional, List
import logging
from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)


class ImageCountFilterProcessor(DatasetProcessor):
    """Filter samples based on the number of images they contain."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_images = config.get("min_images", 0)
        self.max_images = config.get("max_images", float('inf'))
        
        # Fields to check for images
        self.image_fields = config.get("image_fields", ["images", "image"])
        
        # Log configuration
        LOG.info(f"ImageCountFilter initialized with min_images={self.min_images}, max_images={self.max_images}")
    
    def apply_to_dataset(self, dataset):
        """Apply filtering directly to the dataset with detailed logging."""
        initial_count = len(dataset)
        
        def filter_function(example):
            """Filter function that gets applied to each example."""
            image_count = self._count_images(example)
            
            if image_count < self.min_images:
                return False
            
            if image_count > self.max_images:
                return False
                
            return True
        
        LOG.info(f"🔍 ImageCountFilter: Processing {initial_count} examples (min={self.min_images}, max={self.max_images})")
        filtered_dataset = dataset.filter(filter_function)
        final_count = len(filtered_dataset)
        
        # Log detailed results
        filtered_count = initial_count - final_count
        LOG.info(f"📊 ImageCountFilter Results:")
        LOG.info(f"  ✅ Passed: {final_count}/{initial_count} ({final_count/initial_count*100:.1f}%)")
        LOG.info(f"  📉 Total filtered: {filtered_count}/{initial_count} ({filtered_count/initial_count*100:.1f}%)")
        
        return filtered_dataset
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter example based on image count."""
        image_count = self._count_images(example)
        
        LOG.debug(f"ImageCountFilter: example keys = {list(example.keys())}, image_count = {image_count}")
        
        if image_count < self.min_images:
            LOG.debug(f"Filtering example: {image_count} images < min_images ({self.min_images})")
            return None
        
        if image_count > self.max_images:
            LOG.debug(f"Filtering example: {image_count} images > max_images ({self.max_images})")
            return None
        
        LOG.debug(f"ImageCountFilter: example passed with {image_count} images")
        return example
    
    def _count_images(self, example: Dict[str, Any]) -> int:
        """Count the number of images in an example."""
        total_count = 0
        
        for field in self.image_fields:
            if field in example and example[field] is not None:
                value = example[field]
                
                # Handle list of images
                if isinstance(value, list):
                    total_count += len(value)
                # Handle single image
                elif value:
                    total_count += 1
        
        return total_count
    
    def get_required_columns(self) -> List[str]:
        """No hard requirements since image fields might vary."""
        return []


# Register the processor
register_processor("image_count_filter", ImageCountFilterProcessor)