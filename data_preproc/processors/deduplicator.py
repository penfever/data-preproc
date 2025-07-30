"""Deduplicator processor for removing duplicate or similar samples from datasets."""

import multiprocessing as mp
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, Any, Optional, List, Tuple
import logging

from datasets import Dataset, load_dataset
from rapidfuzz import fuzz, process
from tqdm import tqdm

from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)


class DeduplicatorProcessor(DatasetProcessor):
    """Remove duplicate or similar samples using various similarity methods."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.method = config.get("method", "fuzzy")  # fuzzy, ngram, or combined
        self.column = config.get("column", "text")
        self.similarity_threshold = config.get("similarity_threshold", 90.0)
        self.ngram_size = config.get("ngram_size", 8)
        self.external_datasets = config.get("external_datasets", [])
        self.tokenizer = None  # Will be set by the framework
        
        # Validate method
        if self.method not in ["fuzzy", "ngram", "combined"]:
            raise ValueError(f"Invalid deduplication method: {self.method}. Must be 'fuzzy', 'ngram', or 'combined'")
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """This won't be used - we override the dataset processing."""
        return example
    
    def apply_to_dataset(self, dataset):
        """Apply deduplication directly to the dataset."""
        initial_count = len(dataset)
        LOG.info(f"🔍 Deduplicator: Processing {initial_count} examples")
        LOG.info(f"  Method: {self.method}")
        LOG.info(f"  Column: {self.column}")
        LOG.info(f"  Similarity threshold: {self.similarity_threshold}%")
        LOG.info(f"  External datasets: {len(self.external_datasets)}")
        
        # First, perform internal deduplication
        count_after_internal = initial_count
        if self.method == "fuzzy":
            dataset = self._deduplicate_fuzzy(dataset)
        elif self.method == "ngram":
            dataset = self._deduplicate_ngram(dataset)
        elif self.method == "combined":
            dataset = self._deduplicate_combined(dataset)
        
        count_after_internal = len(dataset)
        internal_removed = initial_count - count_after_internal
        
        # Then, if external datasets are specified, deduplicate against them
        if self.external_datasets:
            dataset = self._deduplicate_external(dataset)
        
        final_count = len(dataset)
        external_removed = count_after_internal - final_count
        total_removed = initial_count - final_count
        
        LOG.info(f"📊 Deduplication Results:")
        LOG.info(f"  ✅ Passed: {final_count}/{initial_count} ({final_count/initial_count*100:.1f}%)")
        LOG.info(f"  ❌ Internal duplicates: {internal_removed}")
        LOG.info(f"  ❌ External duplicates: {external_removed}")
        LOG.info(f"  📉 Total removed: {total_removed}/{initial_count} ({total_removed/initial_count*100:.1f}%)")
        
        return dataset
    
    def _deduplicate_fuzzy(self, dataset: Dataset) -> Dataset:
        """Deduplicate using fuzzy string matching."""
        values = [str(x) for x in dataset[self.column] if x is not None]
        unique_values = list(set(values))
        n_processes = mp.cpu_count()
        
        LOG.info(f"Performing fuzzy deduplication on {len(values)} examples (threshold: {self.similarity_threshold})")
        
        process_pair = partial(
            self._fuzz_string_pair,
            values2=unique_values,
            similarity_threshold=self.similarity_threshold,
        )
        
        with Pool(n_processes) as pool:
            all_matches = list(
                tqdm(
                    pool.imap(process_pair, unique_values, chunksize=100),
                    total=len(unique_values),
                    desc="Finding fuzzy duplicates",
                )
            )
        
        # Map strings to their indices
        str_to_indices = defaultdict(list)
        for i, val in enumerate(values):
            str_to_indices[val].append(i)
        
        # Find indices to remove (keep first occurrence)
        indices_to_remove = set()
        for matches_list in all_matches:
            for str1, str2, score in matches_list:
                if score >= self.similarity_threshold and str1 != str2:
                    indices1 = str_to_indices[str1]
                    indices2 = str_to_indices[str2]
                    all_indices = list(set(indices1 + indices2))
                    all_indices.sort()
                    indices_to_remove.update(all_indices[1:])
        
        keep_mask = [i for i in range(len(dataset)) if i not in indices_to_remove]
        clean_dataset = dataset.select(keep_mask)
        
        LOG.info(f"Fuzzy deduplication removed {len(indices_to_remove)} duplicate rows")
        return clean_dataset
    
    def _deduplicate_ngram(self, dataset: Dataset) -> Dataset:
        """Deduplicate using n-gram overlap."""
        if not self.tokenizer:
            LOG.warning("No tokenizer available for n-gram deduplication, falling back to fuzzy method")
            return self._deduplicate_fuzzy(dataset)
        
        values = [str(x) for x in dataset[self.column] if x is not None]
        
        LOG.info(f"Performing n-gram deduplication on {len(values)} examples (n={self.ngram_size})")
        
        # Tokenize all values
        all_tokens = self.tokenizer(values, add_special_tokens=False)["input_ids"]
        
        # Extract n-grams for each text
        all_ngrams = []
        ngram_to_indices = defaultdict(set)
        
        for idx, tokens in enumerate(tqdm(all_tokens, desc="Extracting n-grams")):
            ngrams = self._get_ngrams(tokens, self.ngram_size)
            all_ngrams.append(set(ngrams))
            for ngram in ngrams:
                ngram_to_indices[ngram].add(idx)
        
        # Find duplicate indices based on n-gram overlap
        indices_to_remove = set()
        processed_indices = set()
        
        for idx, ngrams in enumerate(tqdm(all_ngrams, desc="Finding n-gram duplicates")):
            if idx in processed_indices:
                continue
            
            # Find all indices that share n-grams with this one
            similar_indices = set()
            for ngram in ngrams:
                similar_indices.update(ngram_to_indices[ngram])
            
            # Check overlap ratio for each similar index
            for other_idx in similar_indices:
                if other_idx <= idx or other_idx in processed_indices:
                    continue
                
                overlap = len(ngrams & all_ngrams[other_idx])
                overlap_ratio = overlap / min(len(ngrams), len(all_ngrams[other_idx])) * 100
                
                if overlap_ratio >= self.similarity_threshold:
                    indices_to_remove.add(other_idx)
                    processed_indices.add(other_idx)
        
        keep_mask = [i for i in range(len(dataset)) if i not in indices_to_remove]
        clean_dataset = dataset.select(keep_mask)
        
        LOG.info(f"N-gram deduplication removed {len(indices_to_remove)} duplicate rows")
        return clean_dataset
    
    def _deduplicate_combined(self, dataset: Dataset) -> Dataset:
        """Deduplicate using both fuzzy and n-gram methods."""
        LOG.info("Performing combined deduplication (fuzzy + n-gram)")
        
        # First apply fuzzy deduplication
        dataset = self._deduplicate_fuzzy(dataset)
        
        # Then apply n-gram deduplication
        dataset = self._deduplicate_ngram(dataset)
        
        return dataset
    
    def _deduplicate_external(self, dataset: Dataset) -> Dataset:
        """Deduplicate against external datasets."""
        LOG.info(f"Deduplicating against {len(self.external_datasets)} external datasets")
        
        values = [str(x) for x in dataset[self.column] if x is not None]
        indices_to_remove = set()
        
        for ext_config in self.external_datasets:
            ext_path = ext_config.get("path")
            ext_split = ext_config.get("split", "train")
            ext_column = ext_config.get("column", "text")
            ext_subset = ext_config.get("subset")
            ext_data_files = ext_config.get("data_files")
            ext_delimiter = ext_config.get("delimiter")
            
            LOG.info(f"Loading external dataset: {ext_path} (split: {ext_split})")
            if ext_subset:
                LOG.info(f"Using subset: {ext_subset}")
            
            try:
                # Prepare additional kwargs for dataset loading
                load_kwargs = {}
                if ext_data_files:
                    load_kwargs["data_files"] = ext_data_files
                if ext_delimiter:
                    load_kwargs["delimiter"] = ext_delimiter
                
                # Load external dataset with subset support
                if ext_subset:
                    from data_preproc.utils.data import load_dataset_with_subset
                    ext_dataset = load_dataset_with_subset(ext_path, ext_subset, split=ext_split, **load_kwargs)
                else:
                    ext_dataset = load_dataset(ext_path, split=ext_split, **load_kwargs)
                ext_values = [str(x) for x in ext_dataset[ext_column] if x is not None]
                
                # Apply the selected method
                if self.method == "fuzzy":
                    indices = self._find_fuzzy_matches(values, ext_values)
                elif self.method == "ngram":
                    indices = self._find_ngram_matches(values, ext_values)
                elif self.method == "combined":
                    fuzzy_indices = self._find_fuzzy_matches(values, ext_values)
                    ngram_indices = self._find_ngram_matches(values, ext_values)
                    indices = fuzzy_indices | ngram_indices
                
                indices_to_remove.update(indices)
                LOG.info(f"Found {len(indices)} matches with {ext_path}")
                
            except Exception as e:
                LOG.error(f"Failed to load external dataset {ext_path}: {e}")
                continue
        
        keep_mask = [i for i in range(len(dataset)) if i not in indices_to_remove]
        clean_dataset = dataset.select(keep_mask)
        
        LOG.info(f"External deduplication removed {len(indices_to_remove)} rows")
        return clean_dataset
    
    def _find_fuzzy_matches(self, values1: List[str], values2: List[str]) -> set:
        """Find indices in values1 that have fuzzy matches in values2."""
        n_processes = mp.cpu_count()
        indices_to_remove = set()
        
        process_pair = partial(
            self._fuzz_string_pair,
            values2=values2,
            similarity_threshold=self.similarity_threshold,
        )
        
        with Pool(n_processes) as pool:
            matches = list(
                tqdm(
                    pool.imap(process_pair, values1, chunksize=100),
                    total=len(values1),
                    desc="Finding fuzzy matches",
                )
            )
        
        for i, match_list in enumerate(matches):
            if any(score >= self.similarity_threshold for _, _, score in match_list):
                indices_to_remove.add(i)
        
        return indices_to_remove
    
    def _find_ngram_matches(self, values1: List[str], values2: List[str]) -> set:
        """Find indices in values1 that have n-gram matches in values2."""
        if not self.tokenizer:
            LOG.warning("No tokenizer available for n-gram matching")
            return set()
        
        indices_to_remove = set()
        
        # Tokenize and extract n-grams from values2
        tokens2 = self.tokenizer(values2, add_special_tokens=False)["input_ids"]
        all_ngrams2 = set()
        
        for tokens in tqdm(tokens2, desc="Processing external n-grams"):
            all_ngrams2.update(set(self._get_ngrams(tokens, self.ngram_size)))
        
        # Check each value in values1
        tokens1 = self.tokenizer(values1, add_special_tokens=False)["input_ids"]
        
        for idx, tokens in enumerate(tqdm(tokens1, desc="Checking n-gram matches")):
            ngrams = set(self._get_ngrams(tokens, self.ngram_size))
            if ngrams & all_ngrams2:  # If there's any overlap
                indices_to_remove.add(idx)
        
        return indices_to_remove
    
    def _fuzz_string_pair(
        self, str1: str, values2: List[str], similarity_threshold: float
    ) -> List[Tuple]:
        """Find fuzzy matches for a string against a list of strings."""
        matches_with_scores = process.extract(
            str1, values2, scorer=fuzz.ratio, score_cutoff=similarity_threshold
        )
        return [
            (str1, match_tuple[0], match_tuple[1]) for match_tuple in matches_with_scores
        ]
    
    def _get_ngrams(self, tokens: List[int], n: int) -> List[tuple]:
        """Extract n-grams from tokenized text."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    
    def get_required_columns(self) -> List[str]:
        """Return required columns."""
        return [self.column]


# Register the processor
register_processor("deduplicator", DeduplicatorProcessor)