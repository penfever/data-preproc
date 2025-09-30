"""Tests for the image_transform processor."""

import pytest

pytest.importorskip("PIL")

from data_preproc.processors.image_transform import ImageTransformProcessor


def test_resize_mode_validation_on_init():
    """Ensure invalid resize_mode values raise a ValueError."""
    config = {"max_size": [256, 256], "resize_mode": "stretch"}

    with pytest.raises(ValueError, match="Unsupported resize_mode"):
        ImageTransformProcessor(config)


def test_resize_mode_validation_in_transforms():
    """Ensure resize transform configs reject unsupported modes."""
    config = {
        "transforms": [
            {
                "type": "resize",
                "size": 128,
                "mode": "stretch",
            }
        ]
    }

    with pytest.raises(ValueError, match="Unsupported resize mode"):
        ImageTransformProcessor(config)
