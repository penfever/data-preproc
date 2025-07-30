"""Random sampling processor for downsampling/upsampling datasets to a specific size."""

import logging
from typing import Dict, Any, Optional, List
import random

from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)


class RandomSamplerProcessor(DatasetProcessor):
    """Randomly sample a dataset to a specific size with support for both downsampling and upsampling."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sample_size = config.get("sample_size")
        self.seed = config.get("seed", 42)
        self.allow_upsampling = config.get("allow_upsampling", False)
        
        if not self.sample_size or self.sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        
        # Warning about deduplication compatibility
        if self.allow_upsampling:
            LOG.warning("⚠️  RandomSampler with upsampling (sampling with replacement) enabled!")
            LOG.warning("⚠️  Do NOT use deduplication processors after this sampler - they will remove the duplicates created by upsampling!")
        
        LOG.info(f"RandomSampler initialized with sample_size={self.sample_size}, seed={self.seed}, allow_upsampling={self.allow_upsampling}")
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """This won't be used - we override the dataset processing."""
        return example
    
    def apply_to_dataset(self, dataset):
        """Apply random sampling directly to the dataset."""
        initial_count = len(dataset)
        
        LOG.info(f"🔍 RandomSampler: Processing {initial_count} examples")
        LOG.info(f"  Target sample size: {self.sample_size}")
        LOG.info(f"  Random seed: {self.seed}")
        LOG.info(f"  Allow upsampling: {self.allow_upsampling}")
        
        if initial_count == self.sample_size:
            LOG.info(f"📊 RandomSampler Results:")
            LOG.info(f"  ✅ Dataset already exact target size")
            LOG.info(f"  📈 Keeping all {initial_count} examples")
            return dataset
        elif initial_count > self.sample_size:
            # Downsampling: shuffle and select first N examples
            LOG.info(f"  🔽 Downsampling from {initial_count} to {self.sample_size}")
            shuffled_dataset = dataset.shuffle(seed=self.seed)
            sampled_dataset = shuffled_dataset.select(range(self.sample_size))
            
            removed_count = initial_count - self.sample_size
            removal_rate = (removed_count / initial_count * 100)
            
            LOG.info(f"📊 RandomSampler Results:")
            LOG.info(f"  ✅ Downsampled: {self.sample_size}/{initial_count} ({self.sample_size/initial_count*100:.1f}%)")
            LOG.info(f"  📉 Removed: {removed_count}/{initial_count} ({removal_rate:.1f}%)")
            
            return sampled_dataset
        else:
            # Upsampling needed
            if not self.allow_upsampling:
                LOG.warning(f"⚠️  Dataset has {initial_count} examples but target is {self.sample_size}")
                LOG.warning(f"⚠️  Upsampling disabled. Set allow_upsampling=true to enable sampling with replacement")
                LOG.info(f"📊 RandomSampler Results:")
                LOG.info(f"  ✅ Keeping all {initial_count} examples (no upsampling)")
                return dataset
            
            # Upsampling: sample with replacement
            LOG.info(f"  🔼 Upsampling from {initial_count} to {self.sample_size} (sampling with replacement)")
            
            # Set random seed for reproducibility
            random.seed(self.seed)
            
            # Sample indices with replacement
            sampled_indices = [random.randint(0, initial_count - 1) for _ in range(self.sample_size)]
            sampled_dataset = dataset.select(sampled_indices)
            
            added_count = self.sample_size - initial_count
            addition_rate = (added_count / initial_count * 100)
            
            LOG.info(f"📊 RandomSampler Results:")
            LOG.info(f"  ✅ Upsampled: {self.sample_size}/{initial_count} ({self.sample_size/initial_count*100:.1f}%)")
            LOG.info(f"  📈 Added (duplicates): {added_count}/{initial_count} (+{addition_rate:.1f}%)")
            LOG.warning(f"  ⚠️  Dataset now contains duplicates from sampling with replacement!")
            
            return sampled_dataset
    
    def get_required_columns(self) -> List[str]:
        """No required columns for random sampling."""
        return []


# Register the processor
register_processor("random_sampler", RandomSamplerProcessor)