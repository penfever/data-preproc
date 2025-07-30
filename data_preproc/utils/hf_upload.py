"""HuggingFace Hub dataset upload utilities."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

try:
    from datasets import Dataset, DatasetDict, load_from_disk
    from huggingface_hub import HfApi, login, whoami
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from ..utils.logging import get_logger

LOG = get_logger(__name__)


class HFUploadError(Exception):
    """Exception raised for HuggingFace upload errors."""
    pass


def check_hf_auth() -> Optional[str]:
    """
    Check if user is authenticated with HuggingFace Hub.
    
    Returns:
        Username if authenticated, None otherwise.
    """
    if not HAS_DATASETS:
        raise HFUploadError("datasets and huggingface_hub packages required for HF upload")
    
    try:
        user_info = whoami()
        username = user_info["name"]
        LOG.info(f"Authenticated as HuggingFace user: {username}")
        return username
    except Exception as e:
        LOG.warning(f"Not authenticated with HuggingFace Hub: {e}")
        return None


def authenticate_hf(token: Optional[str] = None) -> str:
    """
    Authenticate with HuggingFace Hub.
    
    Args:
        token: HF token. If None, tries to read from HF_TOKEN env var or prompt user.
        
    Returns:
        Username of authenticated user.
        
    Raises:
        HFUploadError: If authentication fails.
    """
    if not HAS_DATASETS:
        raise HFUploadError("datasets and huggingface_hub packages required for HF upload")
    
    # Check if already authenticated
    username = check_hf_auth()
    if username:
        return username
    
    # Try to get token from environment or parameter
    if token is None:
        token = os.getenv("HF_TOKEN")
    
    if token is None:
        raise HFUploadError(
            "HuggingFace token required for upload. Set HF_TOKEN environment variable "
            "or provide token parameter. Get your token at https://huggingface.co/settings/tokens"
        )
    
    try:
        login(token=token)
        user_info = whoami()
        username = user_info["name"]
        LOG.info(f"Successfully authenticated as: {username}")
        return username
    except Exception as e:
        raise HFUploadError(f"HuggingFace authentication failed: {e}")


def generate_dataset_readme(
    dataset_name: str,
    description: str,
    config: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a README.md for the dataset.
    
    Args:
        dataset_name: Name of the dataset
        description: Dataset description
        config: Processing configuration
        stats: Processing statistics
        
    Returns:
        README content as string.
    """
    
    # Get HF upload configuration
    hf_config = config.get("hf_upload", {})
    
    # Create YAML header for dataset card
    yaml_header = "---\n"
    if hf_config.get("license"):
        yaml_header += f"license: {hf_config['license']}\n"
    if hf_config.get("tags"):
        yaml_header += f"tags:\n"
        for tag in hf_config["tags"]:
            yaml_header += f"- {tag}\n"
    yaml_header += "dataset_info:\n"
    yaml_header += f"  dataset_size: {stats.get('processed_samples', 'unknown')} examples\n"
    yaml_header += "---\n\n"
    
    readme_content = yaml_header + f"""# {dataset_name}

{description}

## Dataset Description

This dataset was processed using the [data-preproc](https://github.com/penfever/data-preproc) package for vision-language model training.

### Processing Configuration

- **Base Model**: {config.get('base_model', 'N/A')}
- **Tokenizer**: {config.get('tokenizer_config', 'N/A')}
- **Sequence Length**: {config.get('sequence_len', 'N/A')}
- **Processing Type**: Vision Language (VL)

### Dataset Features

- **input_ids**: Tokenized input sequences
- **attention_mask**: Attention masks for the sequences
- **labels**: Labels for language modeling
- **images**: PIL Image objects
- **messages**: Original conversation messages
- **metadata**: Processing metadata

### Processing Statistics

"""
    
    if stats:
        success_rate = stats.get('success_rate', 'N/A')
        avg_length = stats.get('avg_length', 'N/A')
        truncation_rate = stats.get('truncation_rate', 'N/A')
        
        # Format numeric values
        success_rate_str = f"{success_rate:.1%}" if isinstance(success_rate, (int, float)) else str(success_rate)
        avg_length_str = f"{avg_length:.1f}" if isinstance(avg_length, (int, float)) else str(avg_length)
        truncation_rate_str = f"{truncation_rate:.1%}" if isinstance(truncation_rate, (int, float)) else str(truncation_rate)
        
        readme_content += f"""- **Original Samples**: {stats.get('original_samples', 'N/A')}
- **Processed Samples**: {stats.get('processed_samples', 'N/A')}
- **Success Rate**: {success_rate_str}
- **Average Token Length**: {avg_length_str}
- **Max Token Length**: {stats.get('max_length', 'N/A')}
- **Truncation Rate**: {truncation_rate_str}

"""
    
    readme_content += """### Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-org/your-dataset-name")

# Access samples
sample = dataset["train"][0]
print(f"Input tokens: {len(sample['input_ids'])}")
print(f"Images: {len(sample['images'])}")
print(f"Messages: {sample['messages']}")
```

## License

This dataset is released under the specified license. Please check the license field for details.
"""
    
    return readme_content


def upload_dataset_to_hf(
    dataset_path: Path,
    organization: str,
    dataset_name: str,
    config: Dict[str, Any],
    hf_config: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None
) -> str:
    """
    Upload a processed dataset to HuggingFace Hub.
    
    Args:
        dataset_path: Path to the processed dataset
        organization: HF organization or username
        dataset_name: Name for the dataset on HF Hub
        config: Processing configuration
        hf_config: HF upload configuration
        stats: Processing statistics
        token: HF token for authentication
        
    Returns:
        URL of the uploaded dataset.
        
    Raises:
        HFUploadError: If upload fails.
    """
    if not HAS_DATASETS:
        raise HFUploadError("datasets and huggingface_hub packages required for HF upload")
    
    # Authenticate
    username = authenticate_hf(token)
    
    # Validate paths and data
    if not dataset_path.exists():
        raise HFUploadError(f"Dataset path does not exist: {dataset_path}")
    
    try:
        # Load the dataset
        LOG.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(str(dataset_path))
        
        # Validate dataset structure
        if not isinstance(dataset, Dataset):
            raise HFUploadError(f"Expected Dataset, got {type(dataset)}")
        
        LOG.info(f"Loaded dataset with {len(dataset)} samples")
        
        # Clean up split name if it contains slice notation
        if hasattr(dataset, '_split') and '[' in str(dataset._split):
            # Create a clean copy without the slice notation
            from datasets import Dataset as DatasetClass
            dataset = DatasetClass.from_dict(dataset.to_dict())
        
        # Prepare repository name
        repo_id = f"{organization}/{dataset_name}"
        LOG.info(f"Uploading to HuggingFace Hub: {repo_id}")
        
        # Create dataset description
        description = hf_config.get("description", f"Processed dataset: {dataset_name}")
        
        # Generate README if requested
        readme_content = None
        if hf_config.get("create_readme", True):
            LOG.info("Generating README.md")
            readme_content = generate_dataset_readme(dataset_name, description, config, stats)
        
        # Prepare upload arguments
        push_kwargs = {
            "repo_id": repo_id,
            "private": hf_config.get("private", False),
            "commit_message": hf_config.get("push_to_hub_kwargs", {}).get(
                "commit_message", "Upload processed dataset"
            ),
            "revision": hf_config.get("push_to_hub_kwargs", {}).get("branch", "main"),
        }
        
        # Note: license and tags are handled via the README and dataset card
        # They're not valid arguments for push_to_hub
        
        # Upload the dataset
        LOG.info("Uploading dataset to HuggingFace Hub...")
        dataset.push_to_hub(**push_kwargs)
        
        # Upload README if generated
        if readme_content:
            try:
                api = HfApi()
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                    f.write(readme_content)
                    readme_path = f.name
                
                api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Add dataset README"
                )
                
                # Clean up
                os.unlink(readme_path)
                LOG.info("Uploaded README.md")
                
            except Exception as e:
                LOG.warning(f"Failed to upload README: {e}")
        
        # Upload processing statistics if available
        if stats:
            try:
                api = HfApi()
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(stats, f, indent=2)
                    stats_path = f.name
                
                api.upload_file(
                    path_or_fileobj=stats_path,
                    path_in_repo="processing_stats.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Add processing statistics"
                )
                
                # Clean up
                os.unlink(stats_path)
                LOG.info("Uploaded processing statistics")
                
            except Exception as e:
                LOG.warning(f"Failed to upload statistics: {e}")
        
        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        LOG.info(f"✅ Dataset successfully uploaded to: {dataset_url}")
        
        return dataset_url
        
    except Exception as e:
        raise HFUploadError(f"Failed to upload dataset: {e}")


def should_upload_to_hf(config: Dict[str, Any]) -> bool:
    """
    Check if HuggingFace upload is enabled in configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if upload is enabled and properly configured.
    """
    hf_config = config.get("hf_upload", {})
    
    if not hf_config.get("enabled", False):
        return False
    
    if not hf_config.get("organization") or not hf_config.get("dataset_name"):
        LOG.warning("HF upload enabled but missing organization or dataset_name")
        return False
    
    return True


def validate_hf_config(hf_config: Dict[str, Any]) -> None:
    """
    Validate HuggingFace upload configuration.
    
    Args:
        hf_config: HF upload configuration
        
    Raises:
        HFUploadError: If configuration is invalid.
    """
    required_fields = ["organization", "dataset_name"]
    
    for field in required_fields:
        if not hf_config.get(field):
            raise HFUploadError(f"Missing required HF upload field: {field}")
    
    # Validate dataset name format
    dataset_name = hf_config["dataset_name"]
    if not dataset_name.replace("-", "").replace("_", "").isalnum():
        raise HFUploadError(f"Invalid dataset name: {dataset_name}. Use only letters, numbers, hyphens, and underscores.")
    
    # Validate organization name
    org = hf_config["organization"]
    if not org.replace("-", "").replace("_", "").isalnum():
        raise HFUploadError(f"Invalid organization name: {org}. Use only letters, numbers, hyphens, and underscores.")
    
    LOG.info("HuggingFace upload configuration validated successfully")