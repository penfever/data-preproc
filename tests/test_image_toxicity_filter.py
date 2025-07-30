"""Test image toxicity filter processor functionality."""

import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from datasets import Dataset
from PIL import Image

from data_preproc.processors.image_toxicity_filter import ImageToxicityFilter


class TestImageToxicityFilter:
    """Test cases for ImageToxicityFilter."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create dummy PIL images for testing
        self.safe_image = Image.new('RGB', (100, 100), color='blue')
        self.nsfw_image = Image.new('RGB', (100, 100), color='red')
        self.unsure_image = Image.new('RGB', (100, 100), color='yellow')
        
        # Create test dataset with various image examples
        self.test_data = [
            {"image": self.safe_image, "text": "Safe content", "id": 1},
            {"image": self.nsfw_image, "text": "NSFW content", "id": 2},
            {"image": self.unsure_image, "text": "Unsure content", "id": 3},
            {"images": [self.safe_image, self.nsfw_image], "text": "Mixed content", "id": 4},
            {"text": "No image content", "id": 5},
            {"image": None, "text": "Null image", "id": 6},
        ]
        self.dataset = Dataset.from_list(self.test_data)

    def test_default_configuration(self):
        """Test processor with default configuration."""
        config = {}
        processor = ImageToxicityFilter(config)
        
        # Check default values
        assert processor.model_name == "ViT-B-32"
        assert processor.pretrained == "openai"
        assert processor.nsfw_threshold == 0.3
        assert processor.underage_threshold == 0.3
        assert processor.filter_nsfw is True
        assert processor.filter_unsure is True
        assert processor.filter_underage_risk is True
        assert processor.image_fields == ["image", "images"]

    def test_custom_configuration(self):
        """Test processor with custom configuration."""
        config = {
            "model_name": "ViT-L-14",
            "pretrained": "laion2b_s32b_b82k",
            "nsfw_threshold": 0.2,
            "underage_threshold": 0.15,
            "filter_nsfw": False,
            "filter_unsure": False,
            "filter_underage_risk": True,
            "image_fields": ["custom_image"],
            "log_filtered": True
        }
        processor = ImageToxicityFilter(config)
        
        assert processor.model_name == "ViT-L-14"
        assert processor.pretrained == "laion2b_s32b_b82k"
        assert processor.nsfw_threshold == 0.2
        assert processor.underage_threshold == 0.15
        assert processor.filter_nsfw is False
        assert processor.filter_unsure is False
        assert processor.filter_underage_risk is True
        assert processor.image_fields == ["custom_image"]
        assert processor.log_filtered is True

    def test_custom_safety_categories(self):
        """Test custom safety categories configuration."""
        config = {
            "neutral_categories": ["photo", "picture"],
            "nsfw_categories": ["adult", "explicit"],
            "underage_categories": ["child", "minor"]
        }
        processor = ImageToxicityFilter(config)
        
        assert processor.neutral_categories == ["photo", "picture"]
        assert processor.nsfw_categories == ["adult", "explicit"]
        assert processor.underage_categories == ["child", "minor"]

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    @patch('data_preproc.processors.image_toxicity_filter.ComputeDeviceUtils')
    def test_process_image_safe(self, mock_device_utils, mock_get_model):
        """Test processing safe image."""
        # Mock device utility
        mock_device_utils.move_to_device.side_effect = lambda x, device=None: x
        
        # Mock CLIP model components
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock image processing
        mock_preprocess.return_value = torch.randn(3, 224, 224)
        mock_tokenizer.return_value = torch.randint(0, 1000, (10, 77))
        
        # Mock embeddings - safe image has high similarity with neutral categories
        mock_image_embedding = torch.randn(1, 512)
        mock_model.encode_image.return_value = mock_image_embedding
        
        # Mock text embeddings - first few are neutral, middle are nsfw, last are underage
        mock_text_embeddings = torch.randn(15, 512)  # 12 neutral + 2 nsfw + 1 underage
        mock_model.encode_text.return_value = mock_text_embeddings
        
        # Mock similarities - high similarity with neutral categories
        mock_similarities = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001])
        
        with patch('torch.matmul', return_value=mock_similarities):
            mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
            
            config = {}
            processor = ImageToxicityFilter(config)
            
            safety_tag, embedding = processor._process_image(self.safe_image)
            
            # Should be tagged as safe (UNLIKELY)
            assert safety_tag == "UNLIKELY"
            assert embedding is not None

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    @patch('data_preproc.processors.image_toxicity_filter.ComputeDeviceUtils')
    def test_process_image_nsfw(self, mock_device_utils, mock_get_model):
        """Test processing NSFW image."""
        # Mock device utility
        mock_device_utils.move_to_device.side_effect = lambda x, device=None: x
        
        # Mock CLIP model components
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock image processing
        mock_preprocess.return_value = torch.randn(3, 224, 224)
        mock_tokenizer.return_value = torch.randint(0, 1000, (10, 77))
        
        # Mock embeddings
        mock_image_embedding = torch.randn(1, 512)
        mock_model.encode_image.return_value = mock_image_embedding
        
        mock_text_embeddings = torch.randn(15, 512)
        mock_model.encode_text.return_value = mock_text_embeddings
        
        # Mock similarities - high similarity with NSFW categories (indices 12, 13 are top)
        mock_similarities = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.8, 0.1])
        
        with patch('torch.matmul', return_value=mock_similarities):
            mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
            
            config = {}
            processor = ImageToxicityFilter(config)
            
            safety_tag, embedding = processor._process_image(self.nsfw_image)
            
            # Should be tagged as NSFW (top 2 similarities are NSFW)
            assert safety_tag == "NSFW"
            assert embedding is not None

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    @patch('data_preproc.processors.image_toxicity_filter.ComputeDeviceUtils')
    def test_process_image_unsure(self, mock_device_utils, mock_get_model):
        """Test processing UNSURE image."""
        # Mock device utility
        mock_device_utils.move_to_device.side_effect = lambda x, device=None: x
        
        # Mock CLIP model components
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock image processing
        mock_preprocess.return_value = torch.randn(3, 224, 224)
        mock_tokenizer.return_value = torch.randint(0, 1000, (10, 77))
        
        # Mock embeddings
        mock_image_embedding = torch.randn(1, 512)
        mock_model.encode_image.return_value = mock_image_embedding
        
        mock_text_embeddings = torch.randn(15, 512)
        mock_model.encode_text.return_value = mock_text_embeddings
        
        # Mock similarities - one NSFW category in top 2 (mixed)
        mock_similarities = torch.tensor([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1])
        
        with patch('torch.matmul', return_value=mock_similarities):
            mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
            
            config = {}
            processor = ImageToxicityFilter(config)
            
            safety_tag, embedding = processor._process_image(self.unsure_image)
            
            # Should be tagged as UNSURE (one NSFW in top 2)
            assert safety_tag == "UNSURE"
            assert embedding is not None

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    @patch('data_preproc.processors.image_toxicity_filter.ComputeDeviceUtils')
    def test_process_image_underage_risk(self, mock_device_utils, mock_get_model):
        """Test processing image with underage risk."""
        # Mock device utility
        mock_device_utils.move_to_device.side_effect = lambda x, device=None: x
        
        # Mock CLIP model components
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock image processing
        mock_preprocess.return_value = torch.randn(3, 224, 224)
        mock_tokenizer.return_value = torch.randint(0, 1000, (10, 77))
        
        # Mock embeddings
        mock_image_embedding = torch.randn(1, 512)
        mock_model.encode_image.return_value = mock_image_embedding
        
        mock_text_embeddings = torch.randn(39, 512)  # 12 neutral + 12 nsfw + 15 underage
        mock_model.encode_text.return_value = mock_text_embeddings
        
        # Mock similarities - NSFW but also high underage similarity
        # 12 neutral (low), 12 nsfw (high for indices 24, 25), 15 underage (high for last element)
        mock_similarities = torch.tensor([0.1] * 12 + [0.1] * 10 + [0.9, 0.8] + [0.1] * 14 + [0.7])  # Total 39 elements
        
        with patch('torch.matmul', return_value=mock_similarities):
            with patch('torch.max', return_value=MagicMock(item=lambda: 0.7)):  # High underage similarity
                mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
                
                config = {"underage_threshold": 0.3}
                processor = ImageToxicityFilter(config)
                
                safety_tag, embedding = processor._process_image(self.nsfw_image)
                
                # Should be tagged as UNDERAGE_RISK
                assert safety_tag == "UNDERAGE_RISK"
                assert embedding is not None

    def test_process_image_error_handling(self):
        """Test error handling during image processing."""
        config = {}
        processor = ImageToxicityFilter(config)
        
        # Test with invalid image
        invalid_image = "not an image"
        safety_tag, embedding = processor._process_image(invalid_image)
        
        assert safety_tag is None
        assert embedding is None

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    def test_process_example_safe_image(self, mock_get_model):
        """Test processing example with safe image."""
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
        
        config = {}
        processor = ImageToxicityFilter(config)
        
        # Mock _process_image to return safe result
        with patch.object(processor, '_process_image', return_value=("UNLIKELY", torch.randn(1, 512))):
            example = {"image": self.safe_image, "text": "Safe content"}
            result = processor.process_example(example)
            
            # Should pass through safe content
            assert result == example
            assert processor.stats["passed"] == 1
            assert processor.stats["filtered"] == 0

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    def test_process_example_nsfw_image(self, mock_get_model):
        """Test processing example with NSFW image."""
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
        
        config = {"filter_nsfw": True}
        processor = ImageToxicityFilter(config)
        
        # Mock _process_image to return NSFW result
        with patch.object(processor, '_process_image', return_value=("NSFW", torch.randn(1, 512))):
            example = {"image": self.nsfw_image, "text": "NSFW content"}
            result = processor.process_example(example)
            
            # Should filter out NSFW content
            assert result is None
            assert processor.stats["filtered"] == 1
            assert processor.stats["nsfw_filtered"] == 1

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    def test_process_example_unsure_image(self, mock_get_model):
        """Test processing example with UNSURE image."""
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
        
        config = {"filter_unsure": True}
        processor = ImageToxicityFilter(config)
        
        # Mock _process_image to return UNSURE result
        with patch.object(processor, '_process_image', return_value=("UNSURE", torch.randn(1, 512))):
            example = {"image": self.unsure_image, "text": "Unsure content"}
            result = processor.process_example(example)
            
            # Should filter out UNSURE content when filter_unsure=True
            assert result is None
            assert processor.stats["filtered"] == 1
            assert processor.stats["unsure_filtered"] == 1

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    def test_process_example_unsure_image_allowed(self, mock_get_model):
        """Test processing example with UNSURE image when allowed."""
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
        
        config = {"filter_unsure": False}  # Allow unsure content
        processor = ImageToxicityFilter(config)
        
        # Mock _process_image to return UNSURE result
        with patch.object(processor, '_process_image', return_value=("UNSURE", torch.randn(1, 512))):
            example = {"image": self.unsure_image, "text": "Unsure content"}
            result = processor.process_example(example)
            
            # Should pass through UNSURE content when filter_unsure=False
            assert result == example
            assert processor.stats["passed"] == 1
            assert processor.stats["filtered"] == 0

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    def test_process_example_underage_risk(self, mock_get_model):
        """Test processing example with underage risk."""
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
        
        config = {"filter_underage_risk": True}
        processor = ImageToxicityFilter(config)
        
        # Mock _process_image to return UNDERAGE_RISK result
        with patch.object(processor, '_process_image', return_value=("UNDERAGE_RISK", torch.randn(1, 512))):
            example = {"image": self.nsfw_image, "text": "Risky content"}
            result = processor.process_example(example)
            
            # Should filter out underage risk content
            assert result is None
            assert processor.stats["filtered"] == 1
            assert processor.stats["underage_risk_filtered"] == 1

    def test_process_example_no_images(self):
        """Test processing example with no images."""
        config = {}
        processor = ImageToxicityFilter(config)
        
        example = {"text": "No image content"}
        result = processor.process_example(example)
        
        # Should pass through when no images
        assert result == example
        assert processor.stats["no_images"] == 1
        assert processor.stats["passed"] == 1

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    def test_process_example_multiple_images(self, mock_get_model):
        """Test processing example with multiple images."""
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
        
        config = {"filter_nsfw": True}
        processor = ImageToxicityFilter(config)
        
        # Mock _process_image to return different results for different images
        def mock_process_image(image):
            if image == self.nsfw_image:
                return ("NSFW", torch.randn(1, 512))
            else:
                return ("UNLIKELY", torch.randn(1, 512))
        
        with patch.object(processor, '_process_image', side_effect=mock_process_image):
            example = {"images": [self.safe_image, self.nsfw_image], "text": "Mixed content"}
            result = processor.process_example(example)
            
            # Should filter out if any image is NSFW
            assert result is None
            assert processor.stats["filtered"] == 1
            assert processor.stats["nsfw_filtered"] == 1

    def test_get_required_columns(self):
        """Test that no columns are strictly required."""
        config = {}
        processor = ImageToxicityFilter(config)
        
        required_cols = processor.get_required_columns()
        assert required_cols == []

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    @patch('data_preproc.processors.image_toxicity_filter.ComputeDeviceUtils')
    def test_apply_to_dataset(self, mock_device_utils, mock_get_model):
        """Test applying filter to entire dataset."""
        # Mock device utility
        mock_device_utils.move_to_device.side_effect = lambda x, device=None: x
        mock_device_utils.log_device_info.return_value = None
        
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
        
        config = {"filter_nsfw": True}
        processor = ImageToxicityFilter(config)
        
        # Mock process_example to filter specific examples
        def mock_process_example(example):
            if example.get("id") == 2:  # NSFW content example
                return None  # Filter out
            elif example.get("id") == 4:  # Mixed content with NSFW
                return None  # Filter out
            else:
                return example  # Keep
        
        with patch.object(processor, 'process_example', side_effect=mock_process_example):
            result = processor.apply_to_dataset(self.dataset)
            
            # Should filter out examples with NSFW images (2 examples)
            assert len(result) < len(self.dataset)
            assert len(result) == 4  # 6 - 2 filtered = 4

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        config = {}
        processor = ImageToxicityFilter(config)
        
        # Check initial stats
        assert processor.stats["total_processed"] == 0
        assert processor.stats["filtered"] == 0
        assert processor.stats["passed"] == 0
        assert processor.stats["no_images"] == 0
        assert processor.stats["nsfw_filtered"] == 0
        assert processor.stats["unsure_filtered"] == 0
        assert processor.stats["underage_risk_filtered"] == 0
        assert processor.stats["processing_errors"] == 0
        
        # Process example with no images
        example = {"text": "No image content"}
        processor.process_example(example)
        
        assert processor.stats["total_processed"] == 1
        assert processor.stats["no_images"] == 1
        assert processor.stats["passed"] == 1

    @patch('data_preproc.processors.image_toxicity_filter.get_clip_model')
    def test_different_image_fields(self, mock_get_model):
        """Test processing with different image field names."""
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock()
        mock_get_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)
        
        config = {"image_fields": ["custom_image", "other_images"]}
        processor = ImageToxicityFilter(config)
        
        # Mock _process_image to return safe result
        with patch.object(processor, '_process_image', return_value=("UNLIKELY", torch.randn(1, 512))):
            example = {"custom_image": self.safe_image, "text": "Safe content"}
            result = processor.process_example(example)
            
            # Should process custom field
            assert result == example
            assert processor.stats["passed"] == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])