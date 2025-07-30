# Dataset Processors

The data-preproc package now includes a flexible processor system inspired by LLaMA-Factory and Axolotl. This allows for configurable, maintainable dataset processing without forcing format conversions.

## Overview

Processors are modular components that transform datasets in configurable ways. They follow a registry pattern where processors can be applied in sequence to achieve complex transformations.

## Quick Reference

| Processor | Purpose | Preserves Structure | Key Parameters |
|-----------|---------|-------------------|----------------|
| `passthrough` | No-op testing | ✅ Yes | None |
| `filter` | Text length & field filtering | ✅ Yes | `max_length`, `min_length`, `required_fields` |
| `column_mapping` | Rename columns | ✅ Yes | `column_mapping` |
| `multimodal_filter` | Image/media filtering | ✅ Yes | `max_image_size`, `min_image_size`, `filter_corrupted_images` |
| `qa_to_messages` | Convert Q&A to chat format | ❌ No | `question_field`, `answer_field` |
| `hf_filter` | Token length & image filtering | ✅ Yes | `max_tokens`, `min_tokens`, `max_image_size`, `min_image_size` |
| `image_count_filter` | Filter by number of images | ✅ Yes | `min_images`, `max_images` |
| `advanced_mapping` | Complex field transformations | ✅ Yes | `mappings`, `simple_mappings`, `keep_unmapped` |
| `regex_transform` | Text regex transformations | ❌ No | `transformations`, `default_flags` |
| `image_transform` | Image transformations (resize, crop, etc.) | ❌ No | `transforms`, `image_fields`, `output_format` |
| `regex_filter` | Pattern-based content filtering | ✅ Yes | `filter_patterns`, `logic_mode`, `invert_logic` |
| `deduplicator` | Remove duplicate/similar samples | ✅ Yes | `method`, `column`, `similarity_threshold`, `external_datasets` |
| `text_toxicity_filter` | Filter text content for toxicity | ✅ Yes | `model_type`, `toxicity_threshold`, `check_types`, `filter_mode` |
| `image_toxicity_filter` | Filter images for inappropriate content | ✅ Yes | `model_name`, `nsfw_threshold`, `filter_nsfw`, `filter_unsure` |
| `pipeline` | Compose multiple processors | ✅ Yes | `processors` |

## Available Processors

### Base Processors

#### `passthrough`
Passes data through unchanged (useful for testing).

```yaml
processors:
  - type: passthrough
```

**Parameters:** None

#### `filter`
Filters examples based on various criteria using text length and field requirements.

```yaml
processors:
  - type: filter
    max_length: 10000               # Max characters in text content (optional)
    min_length: 50                  # Min characters in text content (optional)
    required_fields: []             # List of fields that must be present (optional, default: [])
    filter_corrupted_images: false  # Whether to check image validity (optional, default: false)
```

**Parameters:**
- `max_length` (int, optional): Maximum number of characters allowed in text content
- `min_length` (int, optional): Minimum number of characters required in text content
- `required_fields` (list, optional): List of field names that must exist in each example
- `filter_corrupted_images` (bool, optional): Whether to validate images using PIL

#### `column_mapping`
Maps column names to new names.

```yaml
processors:
  - type: column_mapping
    column_mapping:
      old_name: new_name
      question: problem
      instruction: prompt
```

**Parameters:**
- `column_mapping` (dict, required): Dictionary mapping old column names to new names

### Multimodal Processors

#### `multimodal_filter`
Filters multimodal data based on image/video/audio criteria.

```yaml
processors:
  - type: multimodal_filter
    filter_corrupted_images: true    # Remove corrupted images (optional, default: false)
    require_images: false            # Require at least one image (optional, default: false)
    max_image_size: [2048, 2048]     # Max image dimensions [width, height] (optional)
    min_image_size: [32, 32]         # Min image dimensions [width, height] (optional)
```

**Parameters:**
- `filter_corrupted_images` (bool, optional): Validate and remove corrupted images
- `require_images` (bool, optional): Filter out examples without images
- `max_image_size` (list[int, int], optional): Maximum allowed image dimensions [width, height]
- `min_image_size` (list[int, int], optional): Minimum required image dimensions [width, height]

#### `image_count_filter`
Filters examples based on the number of images they contain.

```yaml
processors:
  - type: image_count_filter
    min_images: 3                    # Minimum number of images required (optional, default: 0)
    max_images: 5                    # Maximum number of images allowed (optional, default: inf)
    image_fields: ["images", "image"] # Fields to check for images (optional, default: ["images", "image"])
```

**Parameters:**
- `min_images` (int, optional): Minimum number of images required in each example (default: 0)
- `max_images` (int, optional): Maximum number of images allowed in each example (default: infinity)
- `image_fields` (list, optional): List of field names to check for images (default: ["images", "image"])

#### `qa_to_messages`
Converts Q&A format to conversation messages format.

```yaml
processors:
  - type: qa_to_messages
    question_field: problem          # Primary field for questions (optional, default: "problem")
    answer_field: solution           # Primary field for answers (optional, default: "solution")
    image_field: image               # Primary field for images (optional, default: "image")
    question_alternatives: ["question", "instruction", "input", "original_question"]  # Fallback fields
    answer_alternatives: ["answer", "output", "response", "original_answer"]          # Fallback fields
```

**Parameters:**
- `question_field` (str, optional): Primary field name containing questions
- `answer_field` (str, optional): Primary field name containing answers
- `image_field` (str, optional): Primary field name containing images
- `question_alternatives` (list, optional): Alternative field names to try for questions
- `answer_alternatives` (list, optional): Alternative field names to try for answers

### Special Processors

#### `hf_filter`
Uses HuggingFace Datasets native `.filter()` method to preserve original structure while filtering.

```yaml
processors:
  - type: hf_filter
    max_tokens: 7500                # Max tokenized length (optional)
    min_tokens: 10                  # Min tokenized length (optional)
    filter_corrupted_images: true   # Check image integrity (optional, default: false)
    max_image_size: [2048, 2048]    # Max image dimensions [width, height] (optional)
    min_image_size: [32, 32]        # Min image dimensions [width, height] (optional)
    text_fields: ["problem", "solution", "question", "answer", "text", "content"]  # Fields to tokenize
```

**Parameters:**
- `max_tokens` (int, optional): Maximum number of tokens after tokenization
- `min_tokens` (int, optional): Minimum number of tokens after tokenization
- `filter_corrupted_images` (bool, optional): Validate and remove corrupted images
- `max_image_size` (list[int, int], optional): Maximum allowed image dimensions
- `min_image_size` (list[int, int], optional): Minimum required image dimensions
- `text_fields` (list, optional): List of fields to concatenate and tokenize for length checking

**Note:** The `hf_filter` processor requires a tokenizer to check token lengths. It preserves the original dataset structure and only filters examples.

#### `advanced_mapping`
Performs complex field transformations including nested extraction, list filtering, and multi-target mapping.

```yaml
processors:
  - type: advanced_mapping
    keep_unmapped: true              # Keep fields not mentioned in mappings (optional, default: true)
    remove_source_fields: false      # Remove source fields after mapping (optional, default: false)
    simple_mappings:                 # Simple field renaming (optional)
      old_field: new_field
    mappings:                        # Advanced mapping rules (optional)
      - source: "prompt[0].content"  # Dot notation with list indexing
        target: "question"           # Single target field
      - source: "extra_info.answer"  # Nested field extraction
        targets: ["solution", "original_answer"]  # Multiple target fields
      - source: "prompt"             # List filtering
        filter: {"role": "user"}     # Only items where role="user"
        extract: "content"           # Extract specific field from filtered items
        target: "problem"
      - source: "images[0]"          # First image in list
        extract_first_of: ["bytes", "path"]  # Try fields in order, use first non-null
        target: "image"
```

**Parameters:**
- `keep_unmapped` (bool, optional): Whether to preserve fields not mentioned in mappings (default: true)
- `remove_source_fields` (bool, optional): Remove source fields after mapping (default: false)
- `simple_mappings` (dict, optional): Simple field-to-field renaming
- `mappings` (list, optional): List of advanced mapping rules, each with:
  - `source` (str, required): Source field path using dot notation, supports list indexing
  - `target` or `targets` (str/list): Target field name(s) to populate
  - `filter` (dict, optional): For lists, filter items by field values
  - `extract` (str, optional): Extract specific field from source value
  - `extract_first_of` (list, optional): Try multiple fields in order, use first non-null
  - `take_first` (bool, optional): If result is a list, take first item (default: true)

**Supported Path Syntax:**
- Dot notation: `field.subfield.deepfield`
- List indexing: `field[0]`, `field[-1]` (negative indexing supported)
- Combined: `field[0].subfield`, `list[2].data.value`

#### `regex_transform`
Applies regex transformations to text fields for pattern-based modifications.

```yaml
processors:
  - type: regex_transform
    default_flags: ["MULTILINE", "IGNORECASE"]  # Default regex flags (optional)
    transformations:
      # Transform solution to R1 thinking format
      - field: solution
        pattern: '^(.+?)(?:(?:\n|^)(?:Answer|Final Answer)[:\-\s]*([^\n]+))?$'
        replacement: '<think>\1</think>\n\n<answer>\2</answer>'
        flags: ["MULTILINE", "DOTALL"]
        count: 0  # Replace all occurrences (default: 0)
      
      # Clean up empty answer tags
      - field: solution
        pattern: '<answer></answer>'
        replacement: '<answer>See solution above</answer>'
```

**Parameters:**
- `default_flags` (list, optional): Default regex flags applied to all transformations
- `transformations` (list, required): List of regex transformation rules, each with:
  - `field` (str, required): Text field to transform
  - `pattern` (str, required): Regex pattern to match
  - `replacement` (str, required): Replacement string (supports capture groups)
  - `flags` (list, optional): Regex flags for this transformation
  - `count` (int, optional): Maximum number of replacements (0 = all, default: 0)

**Supported Regex Flags:**
- `IGNORECASE` or `I`: Case-insensitive matching
- `MULTILINE` or `M`: ^ and $ match line boundaries
- `DOTALL` or `S`: . matches newlines
- `VERBOSE` or `X`: Allow comments in regex
- `ASCII` or `A`: ASCII-only matching
- `LOCALE` or `L`: Use locale-aware matching
- `UNICODE` or `U`: Unicode matching

#### `image_transform`
Applies torchvision-style transformations to images including resize, crop, color adjustments, and format conversions.

```yaml
processors:
  - type: image_transform
    image_fields: ["image", "images"]  # Fields containing images (optional, default: ["image"])
    output_format: "pil"              # Output format: "pil" or "tensor" (optional, default: "pil")
    skip_on_error: true               # Skip transformation on error vs filter example (optional, default: true)
    transforms:
      # Resize images to 224x224
      - type: resize
        size: [224, 224]              # [height, width] or int for smallest edge
        interpolation: bilinear       # nearest, bilinear, bicubic, lanczos (optional, default: bilinear)
      
      # Center crop to square
      - type: center_crop
        size: 224                     # int or [height, width]
      
      # Apply color jitter (data augmentation)
      - type: color_jitter
        brightness: 0.2               # 0.0 to 1.0 (optional, default: 0)
        contrast: 0.2                 # 0.0 to 1.0 (optional, default: 0)
        saturation: 0.2               # 0.0 to 1.0 (optional, default: 0)
        hue: 0.1                      # 0.0 to 0.5 (optional, default: 0)
      
      # Convert to grayscale
      - type: grayscale
        num_output_channels: 3        # 1 for grayscale, 3 for RGB grayscale (optional, default: 1)
      
      # Normalize (requires tensor input/output)
      - type: normalize
        mean: [0.485, 0.456, 0.406]   # Per-channel means
        std: [0.229, 0.224, 0.225]    # Per-channel standard deviations
      
      # Convert to tensor format
      - type: to_tensor
      
      # Convert back to PIL
      - type: to_pil
```

**Parameters:**
- `image_fields` (list, optional): List of fields containing images to transform (default: ["image"])
- `output_format` (str, optional): Final output format - "pil" or "tensor" (default: "pil")
- `skip_on_error` (bool, optional): Skip transformation on error vs filter out example (default: true)
- `transforms` (list, required): List of transformations to apply in order

**Supported Transform Types:**
- `resize`: Resize image to specified dimensions
- `center_crop`: Crop image from center to specified size
- `random_crop`: Randomly crop image (deterministic in current implementation)
- `resized_crop`: Crop specific region and resize to target size
- `grayscale`: Convert to grayscale
- `color_jitter`: Adjust brightness, contrast, saturation, hue
- `normalize`: Normalize pixel values with mean/std (requires tensor format)
- `to_tensor`: Convert PIL image to tensor
- `to_pil`: Convert tensor to PIL image

**Dependencies:**
- PIL (required): Basic image operations
- torchvision (optional): Enhanced tensor operations and interpolation
- numpy (optional): Fallback tensor conversions

#### `regex_filter`
Filters examples based on regex pattern matching with support for complex boolean logic and multiple fields.

```yaml
processors:
  - type: regex_filter
    logic_mode: "any"                 # "any" or "all" - how to combine multiple patterns (optional, default: "any")
    invert_logic: false               # Invert the final filtering decision (optional, default: false)
    default_flags: ["MULTILINE", "IGNORECASE"]  # Default regex flags (optional)
    filter_patterns:
      # Remove examples with refusal patterns
      - field: solution
        pattern: '(?i)(sorry|cannot help|unable to|not possible|inappropriate)'
        action: remove                # "remove", "keep", or "keep_only"
        description: "Refusal patterns"  # Optional description for logging
        flags: ["IGNORECASE"]         # Pattern-specific flags (optional)
      
      # Remove examples without mathematical content
      - field: problem
        pattern: '\$.*?\$|\\[.*?\\]'  # LaTeX math notation
        action: keep_only             # Keep only examples that match this
        description: "Must contain math notation"
      
      # Remove examples with inappropriate content
      - field: problem
        pattern: '(?i)(offensive|harmful|violent)'
        action: remove
        description: "Inappropriate content"
      
      # Keep only examples with specific formatting
      - field: solution
        pattern: '<think>.*?</think>'
        action: keep_only
        description: "Must have thinking format"
```

**Parameters:**
- `filter_patterns` (list, required): List of pattern configurations, each with:
  - `field` (str, required): Text field to check
  - `pattern` (str, required): Regex pattern to match
  - `action` (str, required): Action to take - "remove", "keep", or "keep_only"
  - `description` (str, optional): Description for logging and debugging
  - `flags` (list, optional): Regex flags for this specific pattern
- `logic_mode` (str, optional): How to combine multiple patterns - "any" or "all" (default: "any")
- `invert_logic` (bool, optional): Invert the final filtering decision (default: false)
- `default_flags` (list, optional): Default regex flags applied to all patterns

**Pattern Actions:**
- `remove`: Filter out (remove) examples that match this pattern
- `keep`: Keep examples that match this pattern (no effect if no match)
- `keep_only`: Keep ONLY examples that match this pattern (filter out non-matches)

**Logic Modes:**
- `any`: Example is filtered if ANY pattern condition is met
- `all`: Example is filtered only if ALL pattern conditions are met

**Common Use Cases:**
```yaml
# Example 1: Remove refusal responses
processors:
  - type: regex_filter
    filter_patterns:
      - field: solution
        pattern: '(?i)(sorry|cannot|unable|not possible)'
        action: remove
        description: "Remove refusal responses"

# Example 2: Keep only math problems with solutions
processors:
  - type: regex_filter
    logic_mode: "all"  # Must match ALL conditions
    filter_patterns:
      - field: problem
        pattern: '\$.*?\$'  # Has LaTeX math
        action: keep_only
      - field: solution
        pattern: '.{100,}'  # Solution is at least 100 chars
        action: keep_only

# Example 3: Complex content filtering
processors:
  - type: regex_filter
    logic_mode: "any"  # Remove if ANY pattern matches
    filter_patterns:
      # Quality filters
      - field: solution
        pattern: '^.{0,50}$'  # Too short solutions
        action: remove
      - field: problem
        pattern: '(?i)(test|placeholder|example)'
        action: remove
      
      # Content filters  
      - field: solution
        pattern: '(?i)(error|broken|corrupted)'
        action: remove
```

**Advanced Features:**
- **Statistics tracking**: Automatically tracks match and filter rates per pattern
- **Comprehensive logging**: Debug information for pattern matching decisions
- **Flag inheritance**: Default flags can be overridden per pattern
- **Multi-field support**: Different patterns can target different fields
- **Boolean logic**: Flexible combination of multiple conditions

#### `deduplicator`
Removes duplicate or similar samples from datasets using various similarity detection methods.

```yaml
processors:
  - type: deduplicator
    method: "fuzzy"                    # Similarity detection method (required)
    column: "text"                     # Column to check for duplicates (required)
    similarity_threshold: 90.0         # Similarity threshold 0-100 (optional, default: 90.0)
    ngram_size: 8                      # N-gram size for ngram method (optional, default: 8)
    external_datasets:                 # Optional: deduplicate against external datasets
      - path: "HuggingFace/dataset1"
        split: "train"
        column: "content"
      - path: "HuggingFace/dataset2"
        split: "test"
        column: "text"
```

**Parameters:**
- `method` (str, required): Similarity detection method
  - `"fuzzy"`: Character-level similarity using rapidfuzz
  - `"ngram"`: Token-level similarity using n-gram overlap
  - `"combined"`: Apply both fuzzy and n-gram methods sequentially
- `column` (str, required): Column name to check for duplicates
- `similarity_threshold` (float, optional): Similarity threshold from 0-100 (default: 90.0)
- `ngram_size` (int, optional): N-gram size for token-level comparison (default: 8)
- `external_datasets` (list, optional): List of external datasets to deduplicate against, each with:
  - `path` (str): HuggingFace dataset path
  - `split` (str, optional): Dataset split (default: "train")
  - `column` (str, optional): Column to compare against (default: "text")
  - `subset` (str/list, optional): Dataset subset(s) to load (supports single, multiple, or "_ALL")

**Similarity Methods:**
- **Fuzzy matching**: Uses rapidfuzz for character-level similarity detection
  - Fast and effective for exact and near-exact duplicates
  - Good for catching typos and minor variations
  - Threshold: 90+ for strict, 70-89 for moderate, <70 for lenient

- **N-gram overlap**: Uses tokenization and n-gram comparison
  - Semantic similarity detection at token level
  - Better for paraphrases and restructured content
  - Requires tokenizer (automatically provided by framework)
  - Larger n-gram sizes (8-13) are more conservative

- **Combined method**: Applies both fuzzy and n-gram sequentially
  - Most comprehensive deduplication
  - Higher computation cost but better quality
  - Removes both character-level and semantic duplicates

**Usage Examples:**

```yaml
# Basic fuzzy deduplication
processors:
  - type: deduplicator
    method: "fuzzy"
    column: "text"
    similarity_threshold: 85.0

# N-gram deduplication for semantic similarity
processors:
  - type: deduplicator
    method: "ngram"
    column: "problem"
    similarity_threshold: 75.0
    ngram_size: 6

# Combined method for thorough deduplication
processors:
  - type: deduplicator
    method: "combined"
    column: "content"
    similarity_threshold: 90.0
    ngram_size: 8

# External deduplication to prevent test contamination
processors:
  - type: deduplicator
    method: "fuzzy"
    column: "question"
    similarity_threshold: 95.0
    external_datasets:
      - path: "HuggingFaceH4/MATH-500"
        split: "test"
        column: "problem"
      - path: "livecodebench/code_generation_lite"
        split: "test"
        column: "question_content"
      - path: "Idavidrein/gpqa"
        split: "train"
        column: "Question"
        subset: "gpqa_diamond"  # Specific subset support
      - path: "facebook/multilingual_librispeech"
        split: "test"
        column: "text"
        subset: ["spanish", "french"]  # Multiple subsets
```

**Performance Considerations:**
- Uses multiprocessing for efficient large-scale processing
- Memory-efficient batching for large datasets
- Fuzzy method is fastest, n-gram method is slower but more thorough
- External deduplication requires downloading external datasets
- Progress tracking with tqdm for long-running operations

**Dependencies:**
- `rapidfuzz>=3.0.0`: Required for fuzzy string matching
- `transformers`: Tokenizer support (automatically available)
- `datasets`: HuggingFace datasets integration

### Toxicity Filtering

#### `text_toxicity_filter`
Filters text content based on toxicity scores using Detoxify. Supports multiple toxicity types and configurable thresholds.

```yaml
processors:
  - type: text_toxicity_filter
    model_type: "original"             # Detoxify model: "original", "unbiased", "multilingual"
    text_fields: ["problem", "solution", "question", "answer", "text", "content"]  # Fields to check
    
    # Toxicity thresholds (0.0 to 1.0, higher = more toxic)
    toxicity_threshold: 0.7            # Overall toxicity
    severe_toxicity_threshold: 0.5     # Severe toxicity
    obscene_threshold: 0.7             # Obscene content
    threat_threshold: 0.7              # Threatening content
    insult_threshold: 0.7              # Insulting content
    identity_attack_threshold: 0.7     # Identity-based attacks
    sexual_explicit_threshold: 0.7     # Sexually explicit content (multilingual only)
    
    # Which toxicity types to check
    check_types: ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
    
    # Filter mode: "any" (filter if ANY type exceeds) or "all" (filter if ALL types exceed)
    filter_mode: "any"
    
    # Enable logging for debugging
    log_filtered: false
```

**Parameters:**
- `model_type` (str, optional): Detoxify model to use - "original", "unbiased", or "multilingual" (default: "original")
- `text_fields` (list, optional): List of fields to concatenate and check for toxicity (default: ["problem", "solution", "question", "answer", "text", "content"])
- `toxicity_threshold` (float, optional): Threshold for overall toxicity (0.0-1.0, default: 0.7)
- `severe_toxicity_threshold` (float, optional): Threshold for severe toxicity (default: 0.5)
- `obscene_threshold` (float, optional): Threshold for obscene content (default: 0.7)
- `threat_threshold` (float, optional): Threshold for threatening content (default: 0.7)
- `insult_threshold` (float, optional): Threshold for insulting content (default: 0.7)
- `identity_attack_threshold` (float, optional): Threshold for identity-based attacks (default: 0.7)
- `sexual_explicit_threshold` (float, optional): Threshold for sexually explicit content (default: 0.7)
- `check_types` (list, optional): List of toxicity types to check (default: ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"])
- `filter_mode` (str, optional): Logic for combining multiple toxicity types - "any" or "all" (default: "any")
- `log_filtered` (bool, optional): Whether to log filtered examples for debugging (default: false)

**Toxicity Types:**
- `toxicity`: General toxicity score
- `severe_toxicity`: Severe toxicity that could cause harm
- `obscene`: Obscene or vulgar content
- `threat`: Threatening or intimidating language
- `insult`: Insulting or derogatory content
- `identity_attack`: Attacks based on identity characteristics
- `sexual_explicit`: Sexually explicit content (multilingual model only)

**Usage Examples:**

```yaml
# Basic toxicity filtering
processors:
  - type: text_toxicity_filter
    toxicity_threshold: 0.6
    filter_mode: "any"

# Strict safety filtering
processors:
  - type: text_toxicity_filter
    model_type: "multilingual"
    toxicity_threshold: 0.4
    severe_toxicity_threshold: 0.3
    threat_threshold: 0.3
    identity_attack_threshold: 0.3
    check_types: ["toxicity", "severe_toxicity", "threat", "identity_attack"]
    filter_mode: "any"
    log_filtered: true

# Conservative filtering for production use
processors:
  - type: text_toxicity_filter
    model_type: "unbiased"
    toxicity_threshold: 0.5
    severe_toxicity_threshold: 0.3
    obscene_threshold: 0.6
    threat_threshold: 0.4
    insult_threshold: 0.6
    identity_attack_threshold: 0.4
    text_fields: ["question", "answer", "content"]
    filter_mode: "any"
```

#### `image_toxicity_filter`
Filters images for inappropriate content using CLIP-based safety detection following LAION's approach. Detects NSFW content and potential underage risks.

```yaml
processors:
  - type: image_toxicity_filter
    model_name: "ViT-B-32"             # OpenCLIP model to use
    pretrained: "openai"               # Pretrained weights
    image_fields: ["image", "images"]  # Fields containing images
    
    # Safety categories (can be customized)
    neutral_categories: [
      "a photo", "a picture", "a drawing", "an illustration",
      "a painting", "a rendering", "a screenshot", "a selfie",
      "a landscape", "a portrait", "nature", "architecture"
    ]
    
    nsfw_categories: [
      "porn", "pornography", "sex", "sexual content", "nudity",
      "naked", "explicit content", "adult content", "nsfw content",
      "erotic", "xxx", "hardcore"
    ]
    
    underage_categories: [
      "child", "children", "kid", "kids", "minor", "minors",
      "teen", "teenager", "adolescent", "young", "youth",
      "baby", "infant", "toddler", "preteen", "underage"
    ]
    
    # Similarity thresholds (0.0 to 1.0, higher = more similar)
    nsfw_threshold: 0.3                # NSFW similarity threshold
    underage_threshold: 0.3            # Underage similarity threshold
    
    # Filtering settings
    filter_nsfw: true                  # Filter definite NSFW content
    filter_unsure: true                # Filter uncertain/borderline content
    filter_underage_risk: true         # Filter content with underage risk
    
    # Enable logging for debugging
    log_filtered: false
```

**Parameters:**
- `model_name` (str, optional): OpenCLIP model architecture (default: "ViT-B-32")
- `pretrained` (str, optional): Pretrained weights to use (default: "openai")
- `image_fields` (list, optional): List of fields containing images to check (default: ["image", "images"])
- `neutral_categories` (list, optional): List of neutral/safe category descriptions
- `nsfw_categories` (list, optional): List of NSFW category descriptions
- `underage_categories` (list, optional): List of underage-related category descriptions
- `nsfw_threshold` (float, optional): Similarity threshold for NSFW detection (0.0-1.0, default: 0.3)
- `underage_threshold` (float, optional): Similarity threshold for underage detection (0.0-1.0, default: 0.3)
- `filter_nsfw` (bool, optional): Whether to filter definite NSFW content (default: true)
- `filter_unsure` (bool, optional): Whether to filter uncertain/borderline content (default: true)
- `filter_underage_risk` (bool, optional): Whether to filter content with underage risk (default: true)
- `log_filtered` (bool, optional): Whether to log filtered examples for debugging (default: false)

**Safety Tags:**
- `NSFW`: Definite inappropriate content (both top similarities are NSFW)
- `UNSURE`: Uncertain/borderline content (one top similarity is NSFW)
- `UNLIKELY`: Safe content (no NSFW similarities in top results)
- `UNDERAGE_RISK`: Content flagged for potential underage concerns

**Usage Examples:**

```yaml
# Basic image safety filtering
processors:
  - type: image_toxicity_filter
    filter_nsfw: true
    filter_unsure: true
    filter_underage_risk: true

# Conservative filtering for production
processors:
  - type: image_toxicity_filter
    model_name: "ViT-L-14"             # Larger model for better accuracy
    nsfw_threshold: 0.2                # Very conservative
    underage_threshold: 0.2            # Very conservative
    filter_nsfw: true
    filter_unsure: true
    filter_underage_risk: true
    log_filtered: true

# Moderate filtering allowing some uncertainty
processors:
  - type: image_toxicity_filter
    nsfw_threshold: 0.4
    underage_threshold: 0.3
    filter_nsfw: true
    filter_unsure: false              # Allow uncertain content
    filter_underage_risk: true
    image_fields: ["image", "photo", "picture"]

# Custom categories for specific use case
processors:
  - type: image_toxicity_filter
    neutral_categories: [
      "educational content", "scientific diagram", "technical illustration",
      "business presentation", "academic material", "professional photo"
    ]
    nsfw_categories: [
      "adult content", "explicit material", "inappropriate image",
      "sexual content", "nudity", "pornographic material"
    ]
    underage_categories: [
      "child", "minor", "young person", "teenager", "student"
    ]
    nsfw_threshold: 0.25
    underage_threshold: 0.25
    filter_nsfw: true
    filter_unsure: true
    filter_underage_risk: true
```

**Performance Considerations:**
- Uses GPU acceleration if available for faster processing
- Batch processing for efficiency with large datasets
- Model loading is lazy (only when first needed)
- Larger CLIP models (ViT-L-14) provide better accuracy but slower processing
- Conservative thresholds (0.2-0.3) reduce false negatives but increase false positives

**Dependencies:**
- `detoxify>=0.5.0`: Required for text toxicity detection
- `open-clip-torch>=2.20.0`: Required for image toxicity detection
- `torch>=2.0.0`: Required for CLIP model inference
- `PIL`: Required for image processing

**Installation:**
```bash
pip install -e ".[toxicity]"
```

## Modular Processor Features

The processor system supports advanced modular features for enhanced reusability and flexibility:

### Named Processors

Processors can be given names for reuse and reference:

```yaml
processors:
  # Named processor instance
  - name: "strict_filter"
    type: hf_filter
    max_tokens: 2000
    min_tokens: 100
    text_fields: ["problem", "solution"]
  
  # Reuse the same processor type with different config
  - name: "lenient_filter"  
    type: hf_filter
    max_tokens: 8000
    min_tokens: 50
    text_fields: ["question", "answer"]
```

### Conditional Execution

Processors can execute conditionally based on field existence or values:

```yaml
processors:
  # Only process if specific fields exist
  - name: "image_processor"
    type: image_transform
    condition:
      field_exists: ["image", "images"]
    image_fields: ["image", "images"]
    transforms:
      - type: resize
        size: [512, 512]
  
  # Only process if fields don't exist
  - name: "fallback_mapping"
    type: advanced_mapping
    condition:
      field_not_exists: ["formatted_question"]
    mappings:
      - source: "problem"
        target: "formatted_question"
  
  # Only process if field has specific values
  - name: "math_specific_transform"
    type: regex_transform
    condition:
      field_equals:
        category: "mathematics"
        difficulty: "hard"
    transformations:
      - field: "solution"
        pattern: '^(.+)$'
        replacement: 'Mathematical solution: \1'
  
  # Only process if field doesn't have specific values
  - name: "non_trivial_filter"
    type: hf_filter
    condition:
      field_not_equals:
        difficulty: "trivial"
        status: "incomplete"
    max_tokens: 4000
```

**Supported Condition Types:**
- `field_exists`: List of fields that must exist
- `field_not_exists`: List of fields that must not exist  
- `field_equals`: Dictionary of field:value pairs that must match
- `field_not_equals`: Dictionary of field:value pairs that must not match

### Pipeline Processor

Compose multiple processors into reusable sequences:

```yaml
processors:
  # Define some named processors first
  - name: "initial_filter"
    type: hf_filter
    max_tokens: 4000
    min_tokens: 100
  
  - name: "image_check"
    type: image_count_filter
    min_images: 1
    max_images: 1
  
  # Pipeline processor composing multiple steps
  - name: "cleanup_pipeline"
    type: pipeline
    processors:
      # Reference to previously defined processor
      - "initial_filter"
      
      # Inline processor configuration
      - type: multimodal_filter
        filter_corrupted_images: true
        max_image_size: [1024, 1024]
      
      # Another processor reference
      - "image_check"
      
      # Final inline processor
      - name: "final_transform"
        type: advanced_mapping
        condition:
          field_exists: ["problem", "solution"]
        mappings:
          - source: "problem"
            target: "question"
          - source: "solution"
            target: "answer"
```

**Pipeline Features:**
- **Processor References**: Use strings to reference previously defined named processors
- **Inline Configuration**: Define processors directly within the pipeline
- **Sequential Execution**: Processors execute in order, each receiving output of the previous
- **Error Handling**: Pipeline stops on first processor that filters an example
- **Condition Support**: Each processor in pipeline can have its own conditions

### Processor Reuse and References

```yaml
datasets:
  - path: "dataset1"
    processors:
      # Define reusable processors
      - name: "quality_filter"
        type: hf_filter
        max_tokens: 6000
        min_tokens: 200
        filter_corrupted_images: true
      
      - name: "format_converter"
        type: qa_to_messages
        question_field: "problem"
        answer_field: "solution"
  
  - path: "dataset2" 
    processors:
      # Reuse previously defined processors by name
      - "quality_filter"
      - "format_converter"
      
      # Add dataset-specific processing
      - type: image_transform
        condition:
          field_exists: ["image"]
        transforms:
          - type: resize
            size: [224, 224]
```

### Advanced Modular Example

Here's a comprehensive example showing all modular features:

```yaml
processors:
  # Base filters that can be reused
  - name: "base_quality_filter"
    type: hf_filter
    max_tokens: 8000
    min_tokens: 50
    text_fields: ["problem", "solution", "question", "answer"]
  
  - name: "strict_quality_filter"
    type: hf_filter
    max_tokens: 4000
    min_tokens: 200
    text_fields: ["problem", "solution"]
    filter_corrupted_images: true
  
  # Conditional processors for different data types
  - name: "math_formatter"
    type: regex_transform
    condition:
      field_equals:
        subject: "mathematics"
    transformations:
      - field: "solution"
        pattern: '^(.+)$'
        replacement: '<think>\1</think>\n\n<answer>See solution above</answer>'
  
  - name: "image_optimizer"
    type: image_transform
    condition:
      field_exists: ["image"]
    image_fields: ["image"]
    transforms:
      - type: resize
        size: [768, 768]
        interpolation: "lanczos"
  
  # Pipeline for complete processing workflow
  - name: "full_preprocessing_pipeline"
    type: pipeline
    processors:
      # Quality filtering first
      - "base_quality_filter"
      
      # Conditional image processing
      - "image_optimizer"
      
      # Schema transformation
      - type: advanced_mapping
        condition:
          field_exists: ["prompt", "extra_info"]
        mappings:
          - source: "prompt[0].content"
            target: "problem"
          - source: "extra_info.answer"
            target: "solution"
          - source: "images[0]"
            extract_first_of: ["bytes", "path"]
            target: "image"
      
      # Subject-specific formatting
      - "math_formatter"
      
      # Final quality check
      - "strict_quality_filter"
```

**Benefits of Modular Processors:**
- **Reusability**: Define once, use multiple times
- **Maintainability**: Update processor logic in one place
- **Flexibility**: Mix and match processors for different datasets
- **Conditional Logic**: Apply processors only when appropriate
- **Composition**: Build complex workflows from simple components
- **Debugging**: Named processors make logs more readable

## Usage Examples

### Example 1: Basic Filtering

Filter a dataset to remove short examples and corrupted images:

```yaml
datasets:
  - path: my-dataset
    type: vision_language
    processors:
      - type: filter
        min_length: 100
        max_length: 8000
      - type: multimodal_filter
        filter_corrupted_images: true
```

### Example 2: Format Conversion

Convert Q&A format to messages format with filtering:

```yaml
datasets:
  - path: my-qa-dataset
    type: vision_language
    processors:
      - type: multimodal_filter
        filter_corrupted_images: true
        max_image_size: [1024, 1024]
      - type: filter
        min_length: 50
      - type: qa_to_messages
        question_field: problem
        answer_field: solution
```

### Example 3: Column Mapping

Rename columns and filter:

```yaml
datasets:
  - path: my-dataset
    processors:
      - type: column_mapping
        column_mapping:
          input: question
          output: answer
      - type: filter
        required_fields: [question, answer]
```

### Example 4: Filter by Image Count

Filter examples to keep only those with 3-5 images:

```yaml
datasets:
  - path: my-multimodal-dataset
    type: vision_language
    processors:
      - type: image_count_filter
        min_images: 3
        max_images: 5
      - type: multimodal_filter
        filter_corrupted_images: true
```

### Example 5: Advanced Schema Transformation

Transform complex nested schema (e.g., DeepEyes to multimodal-open-r1 format):

```yaml
datasets:
  - path: ChenShawn/DeepEyes-Datasets-47k
    type: vision_language
    processors:
      # First transform the schema
      - type: advanced_mapping
        remove_source_fields: true  # Clean up after transformation
        mappings:
          # Extract user question from prompt list
          - source: "prompt"
            filter: {"role": "user"}
            extract: "content"
            targets: ["problem", "original_question"]
          
          # Extract answer from nested structure
          - source: "extra_info.answer"
            targets: ["solution", "original_answer"]
          
          # Extract first image, trying different fields
          - source: "images[0]"
            extract_first_of: ["bytes", "path"]
            target: "image"
          
          # Copy split info
          - source: "extra_info.split"
            target: "split"
      
      # Then apply filters
      - type: image_count_filter
        min_images: 1
        max_images: 1
      
      - type: hf_filter
        max_tokens: 7000
        text_fields: ["problem", "solution"]
```

## Creating Custom Processors

To create a custom processor:

1. **Create a processor class** inheriting from `DatasetProcessor`:

```python
from data_preproc.processors import DatasetProcessor, register_processor

class MyCustomProcessor(DatasetProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.my_param = config.get("my_param", "default")
    
    def process_example(self, example):
        # Transform example
        example["new_field"] = self.my_param
        return example
    
    def get_required_columns(self):
        return ["required_field"]

# Register the processor
register_processor("my_custom", MyCustomProcessor)
```

2. **Use in configuration**:

```yaml
processors:
  - type: my_custom
    my_param: "custom_value"
```

## Processing Order

Processors are applied in the order specified in the configuration. Each processor receives the output of the previous processor, allowing for complex multi-step transformations.

## Integration with Vision Language Strategy

The vision language strategy automatically handles both:
- **Messages format**: Uses messages directly
- **Q&A format**: Converts on-the-fly during tokenization

This means you can choose whether to convert the format explicitly (using `qa_to_messages` processor) or let the strategy handle it automatically.

## Performance Considerations

- Processors run sequentially and can be chained
- Filtering processors should run early to reduce processing load
- Format conversion processors should run after filtering
- Image processing is memory-intensive, consider batch sizes

## Migration from Automatic Conversion

The old automatic format conversion has been replaced with explicit processors:

**Old (automatic)**:
```yaml
datasets:
  - path: my-dataset
    type: vision_language
    # Format automatically converted
```

**New (explicit)**:
```yaml
datasets:
  - path: my-dataset
    type: vision_language
    processors:
      - type: qa_to_messages  # Only if conversion desired
```

This gives users full control over when and how format conversion happens.