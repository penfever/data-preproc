"""Tests for the HFFilterProcessor token counting logic."""

import pytest
from datasets import Dataset

from data_preproc.processors.hf_filter import HFFilterProcessor


class _StubTokenizer:
    """Simple tokenizer stub that counts whitespace-delimited tokens."""

    name_or_path = "stub-tokenizer"

    def encode(self, text, add_special_tokens=False):  # noqa: D401, D403
        if isinstance(text, str):
            return text.split()
        return []

    def __call__(self, text, add_special_tokens=False):  # noqa: D401, D403
        tokens = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": tokens}


def _build_processor(**config):
    processor = HFFilterProcessor(config)
    processor.tokenizer = _StubTokenizer()
    return processor


def test_token_field_path_filters_pre_tokenized_examples():
    """Examples should be filtered using the configured token field."""
    dataset = Dataset.from_list([
        {"input_ids": [0, 1, 2, 3], "text": "ignored"},
        {"input_ids": [0, 1], "text": "ignored"},
    ])

    processor = _build_processor(min_tokens=3, token_field="input_ids")
    filtered = processor.apply_to_dataset(dataset)

    assert len(filtered) == 1
    assert filtered[0]["input_ids"] == [0, 1, 2, 3]


def test_text_field_fallback_preserved():
    """Legacy text field fallback remains functional."""
    dataset = Dataset.from_list([
        {"text": "short text"},
        {"text": "this example has enough tokens"},
    ])

    processor = _build_processor(min_tokens=4, text_fields=["text"])
    filtered = processor.apply_to_dataset(dataset)

    assert len(filtered) == 1
    assert filtered[0]["text"].startswith("this example")


def test_force_recompute_disables_cache(monkeypatch):
    """Setting force_recompute should disable HF cache usage."""
    dataset = Dataset.from_list([
        {"input_ids": [0, 1, 2], "text": "ignored"},
        {"input_ids": [0, 1, 2, 3, 4], "text": "ignored"},
    ])

    captured = {}
    original_filter = dataset.filter

    def fake_filter(*args, **kwargs):
        captured.update(kwargs)
        return original_filter(*args, **kwargs)

    monkeypatch.setattr(dataset, "filter", fake_filter)

    processor = _build_processor(min_tokens=3, token_field="input_ids", force_recompute=True)
    processor.apply_to_dataset(dataset)

    assert captured.get("load_from_cache_file") is False


def test_force_recompute_default_preserves_cache(monkeypatch):
    """By default we should leave HF caching behaviour untouched."""
    dataset = Dataset.from_list([
        {"input_ids": [0, 1, 2], "text": "ignored"},
        {"input_ids": [0, 1, 2, 3, 4], "text": "ignored"},
    ])

    captured = {}
    original_filter = dataset.filter

    def fake_filter(*args, **kwargs):
        captured.update(kwargs)
        return original_filter(*args, **kwargs)

    monkeypatch.setattr(dataset, "filter", fake_filter)

    processor = _build_processor(min_tokens=3, token_field="input_ids")
    processor.apply_to_dataset(dataset)

    assert "load_from_cache_file" not in captured
