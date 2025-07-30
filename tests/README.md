# Tests

This directory contains test files for the data-preproc package.

## Test Files

- **`test_basic.py`** - Basic functionality and import tests
- **`test_vl_config.py`** - Vision language configuration validation tests  
- **`test_hf_upload.py`** - HuggingFace upload functionality tests

## Running Tests

From the project root directory:

```bash
# Run all tests
python run_tests.py

# Run individual test
python tests/test_basic.py
```

## Test Dependencies

Some tests require optional dependencies:
- `datasets` and `huggingface_hub` for HF upload tests
- `transformers` for tokenizer tests
- `PIL` for image processing tests

Tests will gracefully skip functionality when dependencies are unavailable.