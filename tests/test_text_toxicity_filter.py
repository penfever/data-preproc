"""Test text toxicity filter processor functionality."""

import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset

from data_preproc.processors.text_toxicity_filter import TextToxicityFilter


class TestTextToxicityFilter:
    """Test cases for TextToxicityFilter."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test dataset with various text examples
        self.test_data = [
            {"text": "Hello, this is a nice day!", "id": 1},
            {"text": "I hate this stupid thing", "id": 2},  # Potentially toxic
            {"text": "The weather is beautiful today", "id": 3},
            {"text": "You are an idiot and worthless", "id": 4},  # Toxic
            {"text": "Let's work together on this project", "id": 5},
            {"problem": "What is 2+2?", "solution": "2+2 equals 4", "id": 6},
            {"content": "This is some neutral content", "id": 7},
            {"text": "", "id": 8},  # Empty text
        ]
        self.dataset = Dataset.from_list(self.test_data)

    def test_default_configuration(self):
        """Test processor with default configuration."""
        config = {}
        processor = TextToxicityFilter(config)
        
        # Check default values
        assert processor.model_type == "original"
        assert processor.text_fields == ["problem", "solution", "question", "answer", "text", "content"]
        assert processor.thresholds["toxicity"] == 0.7
        assert processor.filter_mode == "any"
        assert processor.check_types == [
            "toxicity", "severe_toxicity", "obscene", 
            "threat", "insult", "identity_attack"
        ]

    def test_custom_configuration(self):
        """Test processor with custom configuration."""
        config = {
            "model_type": "multilingual",
            "text_fields": ["custom_field"],
            "toxicity_threshold": 0.5,
            "filter_mode": "all",
            "check_types": ["toxicity", "insult"],
            "log_filtered": True
        }
        processor = TextToxicityFilter(config)
        
        assert processor.model_type == "multilingual"
        assert processor.text_fields == ["custom_field"]
        assert processor.thresholds["toxicity"] == 0.5
        assert processor.filter_mode == "all"
        assert processor.check_types == ["toxicity", "insult"]
        assert processor.log_filtered is True

    def test_extract_text_content(self):
        """Test text extraction from examples."""
        config = {"text_fields": ["text", "content"]}
        processor = TextToxicityFilter(config)
        
        # Test single field
        example1 = {"text": "Hello world"}
        assert processor._extract_text_content(example1) == "Hello world"
        
        # Test multiple fields
        example2 = {"text": "Hello", "content": "world"}
        assert processor._extract_text_content(example2) == "Hello world"
        
        # Test empty/missing fields
        example3 = {"other_field": "value"}
        assert processor._extract_text_content(example3) is None
        
        # Test with empty values
        example4 = {"text": "", "content": "world"}
        assert processor._extract_text_content(example4) == "world"

    @patch('data_preproc.processors.text_toxicity_filter.get_detoxify_model')
    def test_process_example_safe_content(self, mock_get_model):
        """Test processing example with safe content."""
        # Mock Detoxify model
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "toxicity": 0.1,
            "severe_toxicity": 0.05,
            "obscene": 0.02,
            "threat": 0.01,
            "insult": 0.03,
            "identity_attack": 0.02
        }
        mock_get_model.return_value = mock_model
        
        config = {"toxicity_threshold": 0.5}
        processor = TextToxicityFilter(config)
        
        example = {"text": "This is safe content"}
        result = processor.process_example(example)
        
        # Should pass through safe content
        assert result == example
        assert processor.stats["passed"] == 1
        assert processor.stats["filtered"] == 0

    @patch('data_preproc.processors.text_toxicity_filter.get_detoxify_model')
    def test_process_example_toxic_content(self, mock_get_model):
        """Test processing example with toxic content."""
        # Mock Detoxify model
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "toxicity": 0.9,  # High toxicity
            "severe_toxicity": 0.1,
            "obscene": 0.2,
            "threat": 0.1,
            "insult": 0.8,  # High insult
            "identity_attack": 0.1
        }
        mock_get_model.return_value = mock_model
        
        config = {"toxicity_threshold": 0.5, "insult_threshold": 0.5}
        processor = TextToxicityFilter(config)
        
        example = {"text": "This is toxic content"}
        result = processor.process_example(example)
        
        # Should filter out toxic content
        assert result is None
        assert processor.stats["filtered"] == 1
        assert processor.stats["passed"] == 0

    @patch('data_preproc.processors.text_toxicity_filter.get_detoxify_model')
    def test_filter_mode_any(self, mock_get_model):
        """Test filter mode 'any' - filter if ANY type exceeds threshold."""
        # Mock Detoxify model
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "toxicity": 0.3,        # Below threshold
            "severe_toxicity": 0.2,  # Below threshold
            "obscene": 0.1,         # Below threshold
            "threat": 0.8,          # ABOVE threshold
            "insult": 0.1,          # Below threshold
            "identity_attack": 0.1   # Below threshold
        }
        mock_get_model.return_value = mock_model
        
        config = {
            "filter_mode": "any",
            "threat_threshold": 0.5,
            "check_types": ["toxicity", "threat", "insult"]
        }
        processor = TextToxicityFilter(config)
        
        example = {"text": "Threatening content"}
        result = processor.process_example(example)
        
        # Should filter because threat exceeds threshold
        assert result is None
        assert processor.stats["filtered"] == 1

    @patch('data_preproc.processors.text_toxicity_filter.get_detoxify_model')
    def test_filter_mode_all(self, mock_get_model):
        """Test filter mode 'all' - filter only if ALL types exceed threshold."""
        # Mock Detoxify model
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "toxicity": 0.8,        # ABOVE threshold
            "severe_toxicity": 0.2,  # Below threshold
            "obscene": 0.1,         # Below threshold
            "threat": 0.8,          # ABOVE threshold
            "insult": 0.8,          # ABOVE threshold
            "identity_attack": 0.1   # Below threshold
        }
        mock_get_model.return_value = mock_model
        
        config = {
            "filter_mode": "all",
            "toxicity_threshold": 0.5,
            "threat_threshold": 0.5,
            "insult_threshold": 0.5,
            "check_types": ["toxicity", "threat", "insult"]
        }
        processor = TextToxicityFilter(config)
        
        example = {"text": "Mixed toxicity content"}
        result = processor.process_example(example)
        
        # Should filter because ALL types exceed threshold
        assert result is None
        assert processor.stats["filtered"] == 1

    def test_process_example_no_text_content(self):
        """Test processing example with no text content."""
        config = {"text_fields": ["missing_field"]}
        processor = TextToxicityFilter(config)
        
        example = {"other_field": "value"}
        result = processor.process_example(example)
        
        # Should pass through (not filter) when no text content
        assert result == example
        assert processor.stats["no_text_content"] == 1
        assert processor.stats["passed"] == 1

    @patch('data_preproc.processors.text_toxicity_filter.get_detoxify_model')
    def test_process_example_error_handling(self, mock_get_model):
        """Test error handling during processing."""
        # Mock Detoxify model to raise exception
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        mock_get_model.return_value = mock_model
        
        config = {}
        processor = TextToxicityFilter(config)
        
        example = {"text": "Some content"}
        result = processor.process_example(example)
        
        # Should pass through on error (graceful degradation)
        assert result == example
        assert processor.stats["passed"] == 1

    @patch('data_preproc.processors.text_toxicity_filter.get_detoxify_model')
    def test_apply_to_dataset(self, mock_get_model):
        """Test applying filter to entire dataset."""
        # Mock Detoxify model
        mock_model = MagicMock()
        def mock_predict(text):
            if "toxic" in text.lower() or "hate" in text.lower():
                return {
                    "toxicity": 0.9,
                    "severe_toxicity": 0.1,
                    "obscene": 0.1,
                    "threat": 0.1,
                    "insult": 0.1,
                    "identity_attack": 0.1
                }
            else:
                return {
                    "toxicity": 0.1,
                    "severe_toxicity": 0.1,
                    "obscene": 0.1,
                    "threat": 0.1,
                    "insult": 0.1,
                    "identity_attack": 0.1
                }
        mock_model.predict.side_effect = mock_predict
        mock_get_model.return_value = mock_model
        
        config = {"toxicity_threshold": 0.5}
        processor = TextToxicityFilter(config)
        
        result = processor.apply_to_dataset(self.dataset)
        
        # Should filter out toxic examples
        assert len(result) < len(self.dataset)
        
        # Check that toxic examples are removed
        remaining_texts = [row.get("text", "") for row in result]
        assert not any("hate" in text.lower() for text in remaining_texts if text)

    def test_get_required_columns(self):
        """Test that no columns are strictly required."""
        config = {}
        processor = TextToxicityFilter(config)
        
        required_cols = processor.get_required_columns()
        assert required_cols == []

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        config = {}
        processor = TextToxicityFilter(config)
        
        # Check initial stats
        assert processor.stats["total_processed"] == 0
        assert processor.stats["filtered"] == 0
        assert processor.stats["passed"] == 0
        assert processor.stats["no_text_content"] == 0
        
        # Process example with no text
        example = {"other_field": "value"}
        processor.process_example(example)
        
        assert processor.stats["total_processed"] == 1
        assert processor.stats["no_text_content"] == 1
        assert processor.stats["passed"] == 1

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = {
            "toxicity_threshold": 0.4,
            "severe_toxicity_threshold": 0.2,
            "obscene_threshold": 0.6,
            "threat_threshold": 0.3,
            "insult_threshold": 0.8,
            "identity_attack_threshold": 0.1,
            "sexual_explicit_threshold": 0.5,
        }
        processor = TextToxicityFilter(config)
        
        assert processor.thresholds["toxicity"] == 0.4
        assert processor.thresholds["severe_toxicity"] == 0.2
        assert processor.thresholds["obscene"] == 0.6
        assert processor.thresholds["threat"] == 0.3
        assert processor.thresholds["insult"] == 0.8
        assert processor.thresholds["identity_attack"] == 0.1
        assert processor.thresholds["sexual_explicit"] == 0.5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])