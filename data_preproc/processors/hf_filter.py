"""HuggingFace Datasets filter processor that preserves original structure."""

import logging
from typing import Any, Dict, List, Optional
from data_preproc.utils.tokenization import token_length_from_value
from collections import Counter

from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class HFFilterProcessor(DatasetProcessor):
    """Apply HuggingFace Datasets .filter() method with tokenization for length checks."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_tokens = config.get("max_tokens")
        self.min_tokens = config.get("min_tokens")
        self.filter_corrupted_images = config.get("filter_corrupted_images", False)
        self.max_image_size = config.get("max_image_size")
        self.min_image_size = config.get("min_image_size")
        self.tokenizer = config.get("tokenizer")  # Will be set by the caller
        self.token_field = config.get("token_field")
        self.force_recompute = config.get("force_recompute", False)
        

        # Text field extraction preferences
        self.text_fields = config.get(
            "text_fields",
            ["problem", "solution", "question", "answer", "text", "content"],
        )

    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """This won't be used - we override the dataset processing."""
        return example

    def apply_to_dataset(self, dataset):
        """Apply filtering directly to the HF dataset using .filter() method."""

        filter_stats = self._init_filter_stats()

        def filter_function(example):
            """Filter function that gets applied to each example."""
            filter_stats['total_processed'] += 1
            
            # Track all failing conditions for this example
            failed = False
            failure_count = 0
            
            # Check token length if tokenizer is available
            if self.tokenizer and (self.max_tokens or self.min_tokens):
                token_count: Optional[int] = None

                if self.token_field:
                    token_count = token_length_from_value(example.get(self.token_field), self.tokenizer)
                    if token_count is None:
                        LOG.debug(
                            "Filtering: no token data found in field '%s'",
                            self.token_field,
                        )
                else:
                    text_content = self._extract_text_content(example)
                    if text_content:
                        tokens = self.tokenizer(text_content, add_special_tokens=False)
                        token_count = len(tokens["input_ids"])

                if token_count is None:
                    LOG.debug("Filtering: no text content found")
                    filter_stats['no_text_content'] += 1
                    failed = True
                    failure_count += 1
                else:
                    if self.max_tokens and token_count > self.max_tokens:
                        LOG.debug(f"Filtering: token count {token_count} > {self.max_tokens}")
                        filter_stats['token_count_too_high'] += 1
                        failed = True
                        failure_count += 1

                    if self.min_tokens and token_count < self.min_tokens:
                        LOG.debug(f"Filtering: token count {token_count} < {self.min_tokens}")
                        filter_stats['token_count_too_low'] += 1
                        failed = True
                        failure_count += 1
            
            # Check image corruption
            if self.filter_corrupted_images and "image" in example:
                if not self._validate_image(example["image"]):
                    LOG.debug("Filtering: corrupted image")
                    filter_stats['corrupted_image'] += 1
                    failed = True
                    failure_count += 1
            
            # Check image size constraints
            if ("image" in example and example["image"] and 
                (self.max_image_size or self.min_image_size)):
                if not self._check_image_size(example["image"]):
                    filter_stats['image_size_invalid'] += 1
                    failed = True
                    failure_count += 1
            
            if not failed:
                filter_stats['passed'] += 1
                return True

            filter_stats['filtered'] += 1
            reasons = self._collect_failure_reasons(example)
            filter_stats["reason_counts"].update(reasons)

            if len(reasons) > 1:
                filter_stats["combo_counts"][tuple(sorted(reasons))] += 1

            return False

        initial_count = len(dataset)
        LOG.info(f"🔍 HF Filter: Processing {initial_count} examples")
        token_source_info: Any
        if self.token_field:
            token_source_info = self.token_field
        else:
            token_source_info = ", ".join(self.text_fields) if self.text_fields else "<none>"
        LOG.info(f"  Token limits: min={self.min_tokens}, max={self.max_tokens}")
        LOG.info(f"  Token source: {token_source_info}")
        LOG.info(f"  Force recompute: {self.force_recompute}")
        LOG.info(f"  Image checks: corrupted={self.filter_corrupted_images}, size_limits={bool(self.max_image_size or self.min_image_size)}")
        
        filter_kwargs = {}
        if self.force_recompute:
            filter_kwargs["load_from_cache_file"] = False

        filtered_dataset = dataset.filter(filter_function, **filter_kwargs)
        final_count = len(filtered_dataset)
        filtered_count = initial_count - final_count

        if filter_stats["total_processed"] == 0 and initial_count > 0:
            LOG.info(
                "  ℹ️ Cached HF filter result reused; recomputing statistics for reporting..."
            )
            filter_stats = self._recompute_filter_stats(dataset)

        self._log_results(initial_count, filtered_count, filter_stats)

        if final_count == 0 and initial_count > 0:
            reason_counts = filter_stats["reason_counts"]
            reasons: List[str] = []
            if reason_counts.get("token_count_too_low"):
                reasons.append(
                    f"token_count_too_low (<{self.min_tokens}): {reason_counts['token_count_too_low']}"
                )
            if reason_counts.get("token_count_too_high"):
                reasons.append(
                    f"token_count_too_high (>{self.max_tokens}): {reason_counts['token_count_too_high']}"
                )
            if reason_counts.get("corrupted_image"):
                reasons.append(f"corrupted_image: {reason_counts['corrupted_image']}")
            if reason_counts.get("image_size_invalid"):
                reasons.append(f"image_size_invalid: {reason_counts['image_size_invalid']}")

            LOG.error(f"⚠️  HF filter removed all examples. Reasons: {'; '.join(reasons)}")
            LOG.error(
                f"Current settings: min_tokens={self.min_tokens}, max_tokens={self.max_tokens}"
            )
            LOG.error(
                "💡 Suggestion: Adjust min_tokens/max_tokens settings based on your dataset's characteristics."
            )

        return filtered_dataset

    def _init_filter_stats(self) -> Dict[str, Any]:
        """Create an empty filter stats structure."""
        return {
            "total_processed": 0,
            "passed": 0,
            "filtered": 0,
            "token_count_too_low": 0,
            "token_count_too_high": 0,
            "no_text_content": 0,
            "corrupted_image": 0,
            "image_size_invalid": 0,
            "multiple_failures": 0,
            "reason_counts": Counter(),
            "combo_counts": Counter(),
        }

    def _collect_failure_reasons(self, example: Dict[str, Any]) -> List[str]:
        """Determine which filter reasons apply to the example."""
        reasons: List[str] = []

        if self.tokenizer and (self.max_tokens or self.min_tokens):
            token_count: Optional[int] = None

            if self.token_field:
                token_count = token_length_from_value(example.get(self.token_field), self.tokenizer)
            else:
                text_content = self._extract_text_content(example)
                if text_content:
                    tokens = self.tokenizer(text_content, add_special_tokens=False)
                    token_count = len(tokens["input_ids"])

            if token_count is None:
                LOG.debug("Filtering: no text content found")
                reasons.append("no_text_content")
            else:
                if self.max_tokens and token_count > self.max_tokens:
                    LOG.debug("Filtering: token count %s > %s", token_count, self.max_tokens)
                    reasons.append("token_count_too_high")

                if self.min_tokens and token_count < self.min_tokens:
                    LOG.debug("Filtering: token count %s < %s", token_count, self.min_tokens)
                    reasons.append("token_count_too_low")

        if self.filter_corrupted_images and "image" in example:
            if not self._validate_image(example["image"]):
                LOG.debug("Filtering: corrupted image")
                reasons.append("corrupted_image")

        if (
            "image" in example
            and example["image"]
            and (self.max_image_size or self.min_image_size)
        ):
            if not self._check_image_size(example["image"]):
                reasons.append("image_size_invalid")

        return reasons

    def _recompute_filter_stats(self, dataset) -> Dict[str, Any]:
        """Re-run filter checks to populate statistics when cache was used."""
        stats = self._init_filter_stats()

        for example in dataset:
            stats["total_processed"] += 1
            reasons = self._collect_failure_reasons(example)

            if not reasons:
                stats["passed"] += 1
                continue

            stats["filtered"] += 1
            stats["reason_counts"].update(reasons)
            if len(reasons) > 1:
                stats["combo_counts"][tuple(sorted(reasons))] += 1

        return stats

    def _log_results(
        self,
        initial_count: int,
        filtered_count: int,
        filter_stats: Dict[str, Any],
    ) -> None:
        """Log aggregated filter results and reason breakdown."""
        LOG.info("📊 HF Filter Results:")
        if initial_count:
            LOG.info(
                "  ✅ Passed: %s/%s (%.1f%%)",
                filter_stats["passed"],
                initial_count,
                filter_stats["passed"] / initial_count * 100,
            )
        else:
            LOG.info("  ✅ Passed: 0/0 (0.0%)")

        reason_counts: Counter = filter_stats["reason_counts"]
        LOG.info(
            "  ❌ No text content found: %s",
            reason_counts.get("no_text_content", 0),
        )
        LOG.info(
            "  ❌ Token count too low (<%s): %s",
            self.min_tokens,
            reason_counts.get("token_count_too_low", 0),
        )
        LOG.info(
            "  ❌ Token count too high (>%s): %s",
            self.max_tokens,
            reason_counts.get("token_count_too_high", 0),
        )
        LOG.info(
            "  ❌ Corrupted images: %s",
            reason_counts.get("corrupted_image", 0),
        )
        LOG.info(
            "  ❌ Invalid image size: %s",
            reason_counts.get("image_size_invalid", 0),
        )

        multiple_failures = sum(filter_stats["combo_counts"].values())
        if multiple_failures > 0:
            LOG.info("  ⚠️  Examples with multiple failures: %s", multiple_failures)

        if reason_counts:
            LOG.info("  ❌ Filtered reason breakdown:")
            for reason, count in sorted(
                reason_counts.items(), key=lambda item: item[1], reverse=True
            ):
                LOG.info("    - %s: %s", self._format_reason_label(reason), count)

        if filter_stats["combo_counts"]:
            LOG.info("  🔁 Combined failure reasons:")
            for combo, count in sorted(
                filter_stats["combo_counts"].items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                combo_labels = ", ".join(
                    self._format_reason_label(reason) for reason in combo
                )
                LOG.info("    - %s: %s", count, combo_labels)

        if initial_count:
            LOG.info(
                "  📉 Total filtered: %s/%s (%.1f%%)",
                filtered_count,
                initial_count,
                filtered_count / initial_count * 100,
            )
        else:
            LOG.info("  📉 Total filtered: 0/0 (0.0%)")

    def _format_reason_label(self, reason: str) -> str:
        """Return a human-readable label for a reason key."""
        mapping = {
            "no_text_content": "No text content found",
            "token_count_too_low": f"Token count too low (<{self.min_tokens})",
            "token_count_too_high": f"Token count too high (>{self.max_tokens})",
            "corrupted_image": "Corrupted images",
            "image_size_invalid": "Invalid image size",
        }
        return mapping.get(reason, reason.replace("_", " ").capitalize())

    def _extract_text_content(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract text content from example for tokenization."""
        texts = []

        for field in self.text_fields:
            if field in example and example[field]:
                texts.append(str(example[field]))

        return " ".join(texts) if texts else None

    def _validate_image(self, image) -> bool:
        """Validate image integrity."""
        if not HAS_PIL:
            return True

        try:
            if hasattr(image, "verify"):
                img_copy = image.copy()
                try:
                    img_copy.verify()
                    return True
                finally:
                    if hasattr(img_copy, "close"):
                        img_copy.close()
                    del img_copy
            return True
        except Exception as exc:  # pragma: no cover - debug logging only
            LOG.debug("Image validation failed: %s", exc)
            return False

    def _check_image_size(self, image) -> bool:
        """Check image size constraints."""
        if not hasattr(image, "size"):
            return True

        width, height = image.size

        if self.max_image_size:
            max_w, max_h = self.max_image_size
            if width > max_w or height > max_h:
                LOG.debug(
                    "Image too large: %sx%s > %sx%s", width, height, max_w, max_h
                )
                return False

        if self.min_image_size:
            min_w, min_h = self.min_image_size
            if width < min_w or height < min_h:
                LOG.debug(
                    "Image too small: %sx%s < %sx%s", width, height, min_w, min_h
                )
                return False

        return True

    def get_required_columns(self) -> List[str]:
        return []


# Register the processor
register_processor("hf_filter", HFFilterProcessor)
