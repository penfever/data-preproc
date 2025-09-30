"""Tests for processing statistics generation."""

import json
from pathlib import Path

import pytest
from datasets import Dataset

from data_preproc.utils.data import _write_processing_stats  # noqa: WPS450 (testing private helper)
from data_preproc.utils.dict import DictDefault


class _StubTokenizer:
    """Minimal tokenizer stub for testing."""

    name_or_path = "stub-tokenizer"

    def encode(self, text, add_special_tokens=False):  # noqa: D401, D403
        # Treat tokens as whitespace-delimited for determinism
        return text.split()


def test_processing_stats_file_contains_histograms(tmp_path):
    """Processing stats should include token and image histograms."""
    data = [
        {"input_ids": [0, 1, 2, 3], "image": {"width": 64, "height": 64}},
        {"input_ids": [5, 6], "image": {"width": 32, "height": 32}},
        {"input_ids": [7, 8, 9], "image": {"width": 64, "height": 64}},
    ]
    dataset = Dataset.from_list(data)

    cfg = DictDefault({
        "base_model": "stub-model",
        "sequence_len": 128,
        "stats_token_field": "input_ids",
    })

    stats_dir = Path(tmp_path)
    _write_processing_stats(stats_dir, cfg, _StubTokenizer(), dataset, None)

    stats_path = stats_dir / "processing_stats.json"
    assert stats_path.exists()

    content = json.loads(stats_path.read_text())

    train_stats = content["datasets"]["train"]
    token_hist = train_stats["token_counts"]["histogram"]
    assert token_hist, "Token histogram should not be empty"

    image_hist = train_stats["image_resolutions"]["histogram"]
    assert image_hist, "Image histogram should not be empty"

    assert any(bucket["resolution"] == "64x64" for bucket in image_hist)


def test_missing_token_field_raises(tmp_path):
    """Expect ValueError if stats_token_field is absent."""
    dataset = Dataset.from_list([{"input_ids": [1, 2, 3]}])
    cfg = DictDefault({})

    stats_dir = Path(tmp_path)

    with pytest.raises(ValueError, match="stats_token_field must be provided"):
        _write_processing_stats(stats_dir, cfg, _StubTokenizer(), dataset, None)
