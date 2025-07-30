# Data-Preproc

Data-Preproc is a lightweight, standalone utility for ingesting, cleaning and transforming datasets, with a particular focus on LLM and VLM fine-tuning.

## Features

- **Dataset Loading**: Support for various dataset formats and sources
- **Prompt Strategies**: Multiple prompt formatting strategies (Alpaca, ChatML, completion, vision-language, etc.)
- **Tokenization**: Efficient tokenization with various strategies
- **Multimodal Support**: Vision-language preprocessing for images, videos, and audio
- **Dataset Processors**: Modular processing pipeline including deduplication, filtering, toxicity detection, and transformations
- **Configuration**: YAML-based configuration system
- **CLI Interface**: Simple command-line interface for preprocessing
- **HuggingFace Integration**: Direct upload to HuggingFace Hub

## Installation

```bash
# Basic installation
pip install -e .

# With vision-language support
pip install -e ".[vision]"

# With development dependencies
pip install -e ".[dev]"

# With toxicity filtering support
pip install -e ".[toxicity]"

# Install all optional dependencies
pip install -e ".[vision,toxicity,dev]"
```

## Quick Start

1. **Install the package:**
   ```bash
   # Basic installation
   pip install -e .
   
   # With vision-language support
   pip install -e ".[vision]"
   ```

2. **Run tests to verify installation:**
   ```bash
   python run_tests.py
   ```

3. **Try a basic example:**
   ```bash
   # Use example configuration
   data-preproc --config configs/example/example_config.yaml
   ```

## Usage

### Command Line

```bash
# Basic preprocessing
data-preproc --config configs/example/example_config.yaml

# Vision-language preprocessing
data-preproc --config configs/example/example_vl_config.yaml

# With custom output path
data-preproc --config config.yaml --dataset_prepared_path ./preprocessed_data

# Debug mode with sample outputs
data-preproc --config config.yaml --debug --debug_num_examples 5

# With HuggingFace upload
data-preproc --config config.yaml --hf_token "your_token"
```

### Configuration

Create a YAML configuration file:

#### Basic Configuration

```yaml
# Base model for tokenizer
base_model: "meta-llama/Llama-2-7b-hf"
tokenizer_config: "meta-llama/Llama-2-7b-hf"

# Sequence length
sequence_len: 2048

# Output directory
dataset_prepared_path: "./data/prepared"

# Dataset configuration
datasets:
  - path: "tatsu-lab/alpaca"
    type: "alpaca"
    
  # Example with subset (configuration)
  - path: "Idavidrein/gpqa"
    type: "alpaca"
    subset: "gpqa_diamond"  # Load specific subset
    
  # Example with multiple subsets (concatenated)
  - path: "facebook/multilingual_librispeech"
    type: "alpaca"
    subset: ["spanish", "french", "german"]  # Load and concatenate multiple subsets
    
  # Example with all subsets (auto-discovered)
  - path: "facebook/multilingual_librispeech"
    type: "alpaca"
    subset: "_ALL"  # Automatically discover and load all available subsets
    
# Training configuration
train_on_inputs: false
batch_size: 4
num_epochs: 1

# Validation set size (optional)
val_set_size: 0.1

# Dataset processors (optional)
processors:
  - type: hf_filter
    max_tokens: 7000
    min_tokens: 100
  - type: deduplicator
    method: "fuzzy"
    column: "text"
    similarity_threshold: 90.0
```

### Vision-Language Configuration

```yaml
# Base model for VL processing
base_model: "microsoft/kosmos-2-patch14-224"
tokenizer_config: "microsoft/kosmos-2-patch14-224"
processor_type: "auto"
processor_config: "microsoft/kosmos-2-patch14-224"

# Sequence length
sequence_len: 2048

# Output directory
dataset_prepared_path: "./data/prepared_vl"

# Vision Language dataset configuration
datasets:
  - path: "sample_vl_data.json"
    type: "vision_language"  # or "vl"
    mm_plugin: "llava"
    image_token: "<image>"
    video_token: "<video>"
    audio_token: "<audio>"

# Training configuration
train_on_inputs: false
batch_size: 2  # Smaller batch for VL data
num_epochs: 1

# Vision processing settings
image_max_pixels: 768
video_fps: 2.0
audio_sampling_rate: 16000
```

### Python API

```python
from data_preproc.core.datasets import load_datasets
from data_preproc.utils.dict import DictDefault

# Load configuration
config = DictDefault({
    "base_model": "meta-llama/Llama-2-7b-hf",
    "sequence_len": 2048,
    "datasets": [
        {"path": "tatsu-lab/alpaca", "type": "alpaca"}
    ]
})

# Load and preprocess datasets
dataset_meta = load_datasets(cfg=config)
train_dataset = dataset_meta.train_dataset
eval_dataset = dataset_meta.eval_dataset
```

#### Vision Language Configuration

For multimodal datasets with images, videos, and audio:

```yaml
# VL model configuration
base_model: "allenai/Molmo-7B-O-0924"
tokenizer_config: "allenai/Molmo-7B-O-0924"
trust_remote_code: true  # Required for VL models

# Longer context for VL
sequence_len: 8192

datasets:
  - path: "lmms-lab/multimodal-open-r1-8k-verified"
    type: "vision_language"
    split: "train[:1000]"  # First 1000 samples
    
    # MM Plugin configuration
    mm_plugin: "base"  # Options: base, llava
    image_token: "<image>"
    video_token: "<video>"
    audio_token: "<audio>"
    
    # Dataset-specific options
    filter_corrupted_images: true
    max_image_size: [1024, 1024]

# Global VL processing configuration
vl_config:
  image_processing:
    verify_images: true
    convert_to_rgb: true
    max_size: [1024, 1024]
    min_size: [32, 32]
  
  video_processing:
    max_frames: 16
    frame_sampling: "uniform"  # uniform, random, keyframes
    target_fps: 1
  
  audio_processing:
    max_duration: 30  # seconds
    sample_rate: 16000
    format: "mel_spectrogram"

# Memory optimization for VL
memory_config:
  use_streaming: false
  prefetch_factor: 2
  num_workers: 4

# VL-specific tokens
special_tokens:
  additional_tokens:
    - "<image>"
    - "<video>"
    - "<audio>"
```

## Supported Dataset Types

- **alpaca**: Standard instruction-following format
- **completion**: Simple text completion/pretraining format
- **chat_template**: Conversation format using tokenizer chat templates
- **vision_language** / **vl**: Multimodal datasets with images, videos, and audio

### Vision Language Support

The `vision_language` strategy supports:

#### Multimodal Plugins (MM Plugins)
- **base**: Generic VL plugin supporting image, video, and audio tokens
- **llava**: LLaVA-specific plugin with enhanced conversation formatting

#### Dataset-Level Configuration
```yaml
datasets:
  - path: "dataset-name"
    type: "vision_language"
    
    # Required MM plugin settings
    mm_plugin: "base"           # Plugin type
    image_token: "<image>"      # Token for images
    video_token: "<video>"      # Token for videos  
    audio_token: "<audio>"      # Token for audio
    
    # Optional dataset-specific settings
    filter_corrupted_images: true        # Filter invalid images
    max_image_size: [1024, 1024]        # Resize large images
    conversation_format: "default"       # Conversation style
```

#### Global VL Configuration
```yaml
vl_config:
  image_processing:
    verify_images: true         # Validate image integrity
    convert_to_rgb: true        # Convert to RGB format
    max_size: [1024, 1024]      # Maximum image dimensions
    min_size: [32, 32]          # Minimum image dimensions
  
  video_processing:
    max_frames: 16              # Maximum frames to extract
    frame_sampling: "uniform"   # Sampling strategy
    target_fps: 1               # Target frames per second
  
  audio_processing:
    max_duration: 30            # Maximum audio length (seconds)
    sample_rate: 16000          # Target sample rate
    format: "mel_spectrogram"   # Audio format
```

#### Example Usage

```python
from data_preproc import load_datasets, get_mm_plugin
from data_preproc.utils.dict import DictDefault

config = DictDefault({
    "base_model": "allenai/Molmo-7B-O-0924",
    "sequence_len": 8192,
    "datasets": [{
        "path": "lmms-lab/multimodal-open-r1-8k-verified",
        "type": "vision_language",
        "mm_plugin": "base",
        "image_token": "<image>"
    }],
    "vl_config": {
        "image_processing": {
            "verify_images": True,
            "max_size": [1024, 1024]
        }
    }
})

# Load and preprocess VL dataset
dataset_meta = load_datasets(cfg=config)
vl_dataset = dataset_meta.train_dataset

# Access VL data
sample = vl_dataset[0]
print(f"Input tokens: {len(sample['input_ids'])}")
print(f"Images: {len(sample['images'])}")
print(f"Messages: {sample['messages']}")
```

### Dataset Processors

The data-preproc package includes a modular processor system for dataset transformations. Processors can be chained together to perform complex data preparation tasks.

#### Available Processors

- **`passthrough`**: No-op testing processor (useful for debugging)
- **`filter`**: Basic text length and field filtering
- **`hf_filter`**: Filter datasets by token length, image constraints, and data quality
- **`column_mapping`**: Simple column renaming
- **`advanced_mapping`**: Transform and map dataset fields with complex transformations
- **`multimodal_filter`**: Filter multimodal content (images, videos, audio)
- **`image_count_filter`**: Filter by number of images in each example
- **`qa_to_messages`**: Convert Q&A format to conversation format
- **`deduplicator`**: Remove duplicate or similar samples using various similarity methods
- **`regex_filter`**: Filter content using regular expressions
- **`regex_transform`**: Apply regex transformations to text fields
- **`image_transform`**: Apply image transformations (resize, crop, normalize)
- **`text_toxicity_filter`**: Filter text content for toxicity using Detoxify
- **`image_toxicity_filter`**: Filter images for inappropriate content using CLIP
- **`pipeline`**: Compose multiple processors into reusable sequences

#### Deduplication

The deduplicator processor removes duplicate or similar samples from datasets:

```yaml
processors:
  - type: deduplicator
    method: "fuzzy"              # Options: "fuzzy", "ngram", "combined"
    column: "text"               # Column to check for duplicates
    similarity_threshold: 90.0   # Similarity threshold (0-100)
    ngram_size: 8               # For n-gram method
    external_datasets:          # Optional: prevent contamination
      - path: "HuggingFaceH4/MATH-500"
        split: "test"
        column: "problem"
        subset: "main"          # Optional: specific subset
      - path: "facebook/multilingual_librispeech"
        split: "test"
        column: "text"
        subset: ["spanish", "french"]  # Optional: multiple subsets
```

**Deduplication Methods:**
- **Fuzzy**: Character-level similarity using rapidfuzz (fast, good for exact duplicates)
- **N-gram**: Token-level similarity using n-gram overlap (semantic similarity)
- **Combined**: Apply both methods for comprehensive deduplication

#### Processing Pipeline Example

```yaml
datasets:
  - path: "my-dataset"
    type: "vision_language"
    processors:
      # 1. Filter by quality
      - type: hf_filter
        max_tokens: 8000
        min_tokens: 100
        filter_corrupted_images: true
      
      # 2. Remove duplicates
      - type: deduplicator
        method: "combined"
        column: "problem"
        similarity_threshold: 85.0
        external_datasets:
          - path: "test-dataset"
            split: "test"
            column: "problem"
      
      # 3. Transform schema
      - type: advanced_mapping
        mappings:
          - source: "question"
            target: "problem"
          - source: "answer"
            target: "solution"
      
      # 4. Convert to messages format
      - type: qa_to_messages
        question_field: "problem"
        answer_field: "solution"
```

For detailed processor documentation, see [PROCESSORS.md](PROCESSORS.md).

### Dataset Subsets

The package supports loading dataset subsets (configurations) for datasets that have multiple configurations:

```yaml
datasets:
  # Single subset
  - path: "Idavidrein/gpqa"
    subset: "gpqa_diamond"
    
  # Multiple subsets (automatically concatenated)
  - path: "facebook/multilingual_librispeech"
    subset: ["spanish", "french", "german"]
    
  # All subsets (automatically discovered and concatenated)
  - path: "facebook/multilingual_librispeech"
    subset: "_ALL"
```

**Subset Features:**
- **Single subset**: Load a specific configuration/subset
- **Multiple subsets**: Load multiple subsets and concatenate them
- **All subsets**: Use `"_ALL"` to automatically discover and load all available subsets
- **Robust error handling**: Gracefully handles missing subsets with warnings
- **Backward compatible**: Works with existing configurations (subset parameter is optional)

### HuggingFace Hub Upload

Automatically upload processed datasets to HuggingFace Hub:

```yaml
# HuggingFace Hub Upload Configuration
hf_upload:
  enabled: true                           # Enable HF upload
  organization: "my-org"                  # HF organization/username
  dataset_name: "my-vl-dataset"          # Dataset name on HF Hub
  private: false                          # Make dataset private
  
  # Optional: Dataset metadata for HF Hub
  description: "Vision-language dataset processed with data-preproc"
  license: "apache-2.0"                  # Dataset license
  tags: ["vision-language", "multimodal", "instruction-following"]
  
  # Upload options
  create_readme: true                     # Generate README.md
  push_to_hub_kwargs:                     # Additional arguments for push_to_hub
    commit_message: "Upload processed VL dataset"
    branch: "main"
```

#### Authentication

Set your HuggingFace token:

```bash
# Environment variable (recommended)
export HF_TOKEN="your_hf_token_here"

# Or via command line
data-preproc --config config.yaml --hf_token "your_token"

# Or in config file (not recommended for security)
hf_upload:
  token: "your_token"  # Not recommended
```

#### Usage Examples

```bash
# Basic preprocessing with HF upload
data-preproc --config configs/example/example_vl_config.yaml

# Preprocessing with custom HF token
data-preproc --config configs/example/example_vl_config.yaml --hf_token hf_your_token

# Override HF settings via CLI
data-preproc --config config.yaml \
  --hf_organization "my-custom-org" \
  --hf_dataset_name "custom-dataset-name"
```

#### Generated Content

When uploading to HuggingFace Hub, the system automatically creates:

- **Dataset Repository**: Complete dataset with all features
- **README.md**: Comprehensive dataset documentation
- **processing_stats.json**: Processing statistics and metadata
- **Dataset Cards**: Proper tagging and description

#### Security Notes

- Never commit HF tokens to version control
- Use environment variables or secure CLI arguments
- Set `private: true` for sensitive datasets
- Review dataset content before uploading

## Project Structure

```
data-preproc/
├── configs/              # Example configuration files
│   ├── example/          # Basic example configs
│   └── test/             # Test configurations
├── data_preproc/         # Main package
│   ├── cli/              # Command-line interface
│   ├── core/             # Core dataset loading functionality
│   ├── loaders/          # Tokenizer and processor loading
│   ├── processors/       # Dataset processors
│   ├── prompt_strategies/ # Prompt formatting strategies
│   ├── utils/            # Utility modules (dict, logging, hf_upload, etc.)
│   ├── mm_plugin.py      # Multimodal plugin system
│   ├── prompters.py      # Prompt template definitions
│   └── prompt_tokenizers.py # Tokenization strategies
├── tests/                # Test suite
├── run_tests.py          # Test runner
├── setup.py              # Package setup
├── pyproject.toml        # Project configuration
├── README.md             # Main documentation
├── PROCESSORS.md         # Detailed processor documentation
└── CLAUDE.md             # Development notes and instructions
```

## License

This project uses the Apache 2.0 license.