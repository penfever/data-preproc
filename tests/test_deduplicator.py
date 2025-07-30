"""Test deduplicator processor functionality."""

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from data_preproc.processors.deduplicator import DeduplicatorProcessor


class TestDeduplicatorProcessor:
    """Test cases for DeduplicatorProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Create test dataset with duplicates
        self.test_data = [
            {"text": "This is a test sentence.", "id": 1},
            {"text": "This is a test sentence.", "id": 2},  # Exact duplicate
            {"text": "This is a test sentence!", "id": 3},  # Very similar
            {"text": "A completely different sentence.", "id": 4},
            {"text": "Another unique sentence here.", "id": 5},
            {"text": "This is a test sentence?", "id": 6},  # Similar with different punctuation
            {"text": "Completely different content.", "id": 7},
        ]
        self.dataset = Dataset.from_list(self.test_data)

    def test_fuzzy_deduplication(self):
        """Test fuzzy deduplication method."""
        config = {
            "method": "fuzzy",
            "column": "text",
            "similarity_threshold": 90.0
        }
        
        processor = DeduplicatorProcessor(config)
        processor.tokenizer = self.tokenizer
        
        result = processor.apply_to_dataset(self.dataset)
        
        # Should remove duplicates and very similar texts
        assert len(result) < len(self.dataset)
        assert len(result) >= 3  # Should keep at least 3 unique texts
        
        # Check that exact duplicates are removed
        texts = [row["text"] for row in result]
        assert len(set(texts)) == len(texts)  # All remaining texts should be unique

    def test_ngram_deduplication(self):
        """Test n-gram deduplication method."""
        config = {
            "method": "ngram",
            "column": "text",
            "similarity_threshold": 80.0,
            "ngram_size": 3
        }
        
        processor = DeduplicatorProcessor(config)
        processor.tokenizer = self.tokenizer
        
        result = processor.apply_to_dataset(self.dataset)
        
        # Should remove duplicates based on n-gram overlap
        assert len(result) < len(self.dataset)
        assert len(result) >= 3  # Should keep at least 3 unique texts

    def test_combined_deduplication(self):
        """Test combined deduplication method."""
        config = {
            "method": "combined",
            "column": "text",
            "similarity_threshold": 85.0,
            "ngram_size": 4
        }
        
        processor = DeduplicatorProcessor(config)
        processor.tokenizer = self.tokenizer
        
        result = processor.apply_to_dataset(self.dataset)
        
        # Should be most aggressive in removing duplicates
        assert len(result) < len(self.dataset)
        assert len(result) >= 3  # Should keep at least 3 unique texts

    def test_no_tokenizer_fallback(self):
        """Test that n-gram method falls back to fuzzy when no tokenizer is available."""
        config = {
            "method": "ngram",
            "column": "text",
            "similarity_threshold": 90.0
        }
        
        processor = DeduplicatorProcessor(config)
        # Don't set tokenizer
        
        result = processor.apply_to_dataset(self.dataset)
        
        # Should still work by falling back to fuzzy method
        assert len(result) < len(self.dataset)

    def test_different_columns(self):
        """Test deduplication on different columns."""
        # Create dataset with duplicates in different column
        data = [
            {"content": "First content", "id": 1},
            {"content": "First content", "id": 2},  # Duplicate
            {"content": "Second content", "id": 3},
        ]
        dataset = Dataset.from_list(data)
        
        config = {
            "method": "fuzzy",
            "column": "content",
            "similarity_threshold": 95.0
        }
        
        processor = DeduplicatorProcessor(config)
        processor.tokenizer = self.tokenizer
        
        result = processor.apply_to_dataset(dataset)
        
        # Should remove the duplicate
        assert len(result) == 2

    def test_threshold_sensitivity(self):
        """Test that different thresholds produce different results."""
        # Test with high threshold (strict)
        config_strict = {
            "method": "fuzzy",
            "column": "text",
            "similarity_threshold": 95.0
        }
        
        processor_strict = DeduplicatorProcessor(config_strict)
        processor_strict.tokenizer = self.tokenizer
        result_strict = processor_strict.apply_to_dataset(self.dataset)
        
        # Test with low threshold (lenient)
        config_lenient = {
            "method": "fuzzy",
            "column": "text",
            "similarity_threshold": 70.0
        }
        
        processor_lenient = DeduplicatorProcessor(config_lenient)
        processor_lenient.tokenizer = self.tokenizer
        result_lenient = processor_lenient.apply_to_dataset(self.dataset)
        
        # Lenient should remove more samples
        assert len(result_lenient) <= len(result_strict)

    def test_empty_dataset(self):
        """Test behavior with empty dataset."""
        empty_dataset = Dataset.from_list([])
        
        config = {
            "method": "fuzzy",
            "column": "text",
            "similarity_threshold": 90.0
        }
        
        processor = DeduplicatorProcessor(config)
        processor.tokenizer = self.tokenizer
        
        result = processor.apply_to_dataset(empty_dataset)
        
        # Should handle empty dataset gracefully
        assert len(result) == 0

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        config = {
            "method": "invalid_method",
            "column": "text",
            "similarity_threshold": 90.0
        }
        
        with pytest.raises(ValueError, match="Invalid deduplication method"):
            DeduplicatorProcessor(config)

    def test_missing_column(self):
        """Test behavior when specified column is missing."""
        config = {
            "method": "fuzzy",
            "column": "missing_column",
            "similarity_threshold": 90.0
        }
        
        processor = DeduplicatorProcessor(config)
        processor.tokenizer = self.tokenizer
        
        # Should raise an error when trying to access missing column
        with pytest.raises(KeyError):
            processor.apply_to_dataset(self.dataset)

    def test_required_columns(self):
        """Test that required columns are correctly identified."""
        config = {
            "method": "fuzzy",
            "column": "text",
            "similarity_threshold": 90.0
        }
        
        processor = DeduplicatorProcessor(config)
        required_cols = processor.get_required_columns()
        
        assert required_cols == ["text"]

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])