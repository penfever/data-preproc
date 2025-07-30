"""Test subset support in dataset loading."""

import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset

from data_preproc.utils.data import load_dataset_with_subset


class TestSubsetSupport:
    """Test cases for dataset subset support."""

    def test_load_single_subset(self):
        """Test loading a single subset."""
        mock_dataset = Dataset.from_dict({"text": ["example1", "example2"]})
        
        with patch("data_preproc.utils.data.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset
            
            result = load_dataset_with_subset("test_dataset", "subset1", split="train")
            
            # Should call load_dataset with subset parameter
            mock_load.assert_called_once_with(
                "test_dataset", "subset1", split="train", streaming=False
            )
            assert result == mock_dataset

    def test_load_multiple_subsets(self):
        """Test loading multiple subsets."""
        mock_dataset1 = Dataset.from_dict({"text": ["example1", "example2"]})
        mock_dataset2 = Dataset.from_dict({"text": ["example3", "example4"]})
        
        with patch("data_preproc.utils.data.load_dataset") as mock_load:
            mock_load.side_effect = [mock_dataset1, mock_dataset2]
            
            with patch("data_preproc.utils.data.concatenate_datasets") as mock_concat:
                mock_concat.return_value = Dataset.from_dict({"text": ["example1", "example2", "example3", "example4"]})
                
                result = load_dataset_with_subset("test_dataset", ["subset1", "subset2"], split="train")
                
                # Should call load_dataset twice
                assert mock_load.call_count == 2
                mock_concat.assert_called_once_with([mock_dataset1, mock_dataset2])

    def test_load_all_subsets(self):
        """Test loading all subsets with _ALL parameter."""
        mock_dataset1 = Dataset.from_dict({"text": ["example1", "example2"]})
        mock_dataset2 = Dataset.from_dict({"text": ["example3", "example4"]})
        
        with patch("data_preproc.utils.data.get_dataset_config_names") as mock_get_configs:
            mock_get_configs.return_value = ["subset1", "subset2"]
            
            with patch("data_preproc.utils.data.load_dataset") as mock_load:
                mock_load.side_effect = [mock_dataset1, mock_dataset2]
                
                with patch("data_preproc.utils.data.concatenate_datasets") as mock_concat:
                    mock_concat.return_value = Dataset.from_dict({"text": ["example1", "example2", "example3", "example4"]})
                    
                    result = load_dataset_with_subset("test_dataset", "_ALL", split="train")
                    
                    # Should discover subsets and load both
                    mock_get_configs.assert_called_once_with("test_dataset")
                    assert mock_load.call_count == 2
                    mock_concat.assert_called_once_with([mock_dataset1, mock_dataset2])

    def test_all_subsets_no_configs_found(self):
        """Test _ALL when no subsets are found."""
        mock_dataset = Dataset.from_dict({"text": ["example1", "example2"]})
        
        with patch("data_preproc.utils.data.get_dataset_config_names") as mock_get_configs:
            mock_get_configs.return_value = []  # No subsets found
            
            with patch("data_preproc.utils.data.load_dataset") as mock_load:
                mock_load.return_value = mock_dataset
                
                result = load_dataset_with_subset("test_dataset", "_ALL", split="train")
                
                # Should fall back to loading without subset
                mock_load.assert_called_once_with(
                    "test_dataset", split="train", streaming=False
                )

    def test_subset_loading_error_handling(self):
        """Test error handling when subset loading fails."""
        mock_dataset = Dataset.from_dict({"text": ["example1", "example2"]})
        
        with patch("data_preproc.utils.data.load_dataset") as mock_load:
            # First call fails, second succeeds
            mock_load.side_effect = [Exception("Subset not found"), mock_dataset]
            
            # Should try to load without subset after subset fails
            result = load_dataset_with_subset("test_dataset", "nonexistent_subset", split="train")
            
            # Should call load_dataset twice: once with subset (fails), once without (succeeds)
            assert mock_load.call_count == 2
            assert result == mock_dataset

    def test_config_discovery_error_handling(self):
        """Test error handling when config discovery fails."""
        mock_dataset = Dataset.from_dict({"text": ["example1", "example2"]})
        
        with patch("data_preproc.utils.data.get_dataset_config_names") as mock_get_configs:
            mock_get_configs.side_effect = Exception("Config discovery failed")
            
            with patch("data_preproc.utils.data.load_dataset") as mock_load:
                mock_load.return_value = mock_dataset
                
                result = load_dataset_with_subset("test_dataset", "_ALL", split="train")
                
                # Should fall back to loading without subset
                mock_load.assert_called_once_with(
                    "test_dataset", split="train", streaming=False
                )

    def test_with_data_files(self):
        """Test subset loading with data_files parameter."""
        mock_dataset = Dataset.from_dict({"text": ["example1", "example2"]})
        
        with patch("data_preproc.utils.data.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset
            
            result = load_dataset_with_subset(
                "test_dataset", "subset1", split="train", data_files=["file1.json", "file2.json"]
            )
            
            # Should call load_dataset with data_files parameter
            mock_load.assert_called_once_with(
                "test_dataset", "subset1", data_files=["file1.json", "file2.json"], 
                split="train", streaming=False
            )

    def test_with_streaming(self):
        """Test subset loading with streaming mode."""
        mock_dataset = Dataset.from_dict({"text": ["example1", "example2"]})
        
        with patch("data_preproc.utils.data.load_dataset") as mock_load:
            mock_load.return_value = mock_dataset
            
            result = load_dataset_with_subset("test_dataset", "subset1", split="train", streaming=True)
            
            # Should call load_dataset with streaming=True
            mock_load.assert_called_once_with(
                "test_dataset", "subset1", split="train", streaming=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])